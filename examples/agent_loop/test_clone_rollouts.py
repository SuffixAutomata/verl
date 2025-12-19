"""
Smoke test for the clone-aware ToolAgentLoop.

This runs a single rollout with a pretrained HF model (no training) where the
root calls the built-in `spawn_clone` tool with multiple objectives and we
verify that multiple rollouts (root + clones) are returned in the batch.

Notes:
- This script does not start any Ray servers; it uses a local HF model through
  a minimal server manager shim.
- The model you choose must be able to emit tool calls in the parser format you
  configure (default: Hermes-style <tool_call>...</tool_call>).
- No tests are executed here; run this script in an environment with the model
  available.
"""

import argparse
import asyncio
import os
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import _DummyConfig
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.workers.rollout.replica import TokenOutput


class LocalHFServerManager:
    """Thin async wrapper that matches AsyncLLMServerManager.generate signature."""

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data=None,
    ) -> TokenOutput:
        loop = asyncio.get_running_loop()
        max_new_tokens = int(sampling_params.get("max_new_tokens", 512))
        temperature = float(sampling_params.get("temperature", 0.7))
        top_p = float(sampling_params.get("top_p", 0.9))

        input_ids = torch.tensor(prompt_ids, device=self.device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        def _generate_sync():
            with torch.no_grad():
                print(input_ids, attention_mask, max_new_tokens, temperature, top_p)
                return self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.model.config.eos_token_id,
                )

        output_ids = await loop.run_in_executor(None, _generate_sync)
        new_tokens = output_ids[0, input_ids.shape[1] :].tolist()
        print(f"Generated {len(new_tokens)} tokens")
        return TokenOutput(token_ids=new_tokens)


def build_minimal_config(model_path: str) -> Any:
    """Return a minimal OmegaConf matching ToolAgentLoop expectations."""
    cfg = {
        "trainer": {"project_name": "clone_smoke", "experiment_name": "local"},
        "data": {"apply_chat_template_kwargs": {}},
        "actor_rollout_ref": {
            "model": {"path": model_path, "custom_chat_template": None},
            "rollout": {
                "prompt_length": 2048,
                "response_length": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
                "calculate_log_probs": False,
                "multi_turn": {
                    "enable": True,
                    "format": "hermes",  # or "gpt-oss" to match your model
                    "max_user_turns": 4,
                    "max_assistant_turns": 4,
                    "max_parallel_calls": 4,
                    "max_tool_response_length": 512,
                    "tool_response_truncate_side": "right",
                    "tool_config_path": None,  # add if you want extra tools
                    "interaction_config_path": None,
                    "clone": {
                        "enable": True,
                        "max_clones_per_call": 4,
                        "max_clone_depth": 1,
                        "allow_tool_use": True,
                        "allow_nested_clones": False,
                        "reward_function": None,  # e.g. "my_pkg.rewards:clone_reward_fn"
                    },
                },
                "agent": {"num_workers": 1, "default_agent_loop": "tool_agent"},
                "trace": {},
            },
        },
        "reward_model": {
            "enable": False,
            "enable_resource_pool": False,
            "use_reward_loop": False,
        },
        "trainer_config": {},
    }
    return OmegaConf.create(cfg)


def build_prompt(tasks: list[str]) -> list[dict[str, str]]:
    """Construct a prompt that strongly nudges the model to call spawn_clone."""
    task_list = "\\n- ".join(tasks)
    instructions = (
        "You are the root agent. Call the tool `spawn_clone` with all objectives below and nothing else. "
        "Use Hermes-style tool calls: <tool_call>{\"name\": \"spawn_clone\", "
        "\"arguments\": {\"objectives\": [\"task1\", \"task2\"]}}</tool_call>. "
        "After the tool responses are processed you will see the summaries from clones."
    )
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"Objectives to delegate:\n- {task_list}"},
    ]


async def main(args):
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if "cuda" in device.type else torch.float32,
        device_map="auto" if "cuda" in device.type else None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = build_minimal_config(args.model)
    server_manager = LocalHFServerManager(model, device)

    # Prepare agent loop
    agent_loop = ToolAgentLoop(
        trainer_config=_DummyConfig(config=config),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
    )

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "logprobs": False,
    }

    prompt = build_prompt(args.tasks)
    outputs = await agent_loop.run(
        sampling_params,
        raw_prompt=prompt,
        multi_modal_data={},
        tools_kwargs={},
        extra_info={"interaction_kwargs": {}},
    )

    outputs_list = outputs if isinstance(outputs, list) else [outputs]
    print(f"Returned {len(outputs_list)} rollouts")
    for idx, out in enumerate(outputs_list):
        role = out.extra_fields.get("actor_role", "unknown")
        clone_id = out.extra_fields.get("clone_id")
        label = out.extra_fields.get("clone_label")
        reward = out.reward_score
        text = tokenizer.decode(out.response_ids, skip_special_tokens=True)
        print(f"[{idx}] role={role} clone_id={clone_id} label={label} reward={reward}")
        print(text)
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HF model name or path.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["list three facts about the riemann hypothesis", "list three facts about shakespeare's hamlet"],
        help="Objectives to pass to spawn_clone.",
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    asyncio.run(main(args))
