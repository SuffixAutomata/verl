"""
Smoke test for the clone-aware ToolAgentLoop using a vLLM rollout server.

This script composes a full Hydra config (so rollout configs carry _target_),
reads a test parquet dataset, and runs each prompt through ToolAgentLoop.
Rewards come from the clone reward function configured in the rollout config.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Iterable

import datasets
import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager, _DummyConfig
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.workers.rollout.replica import get_rollout_replica_class

DEFAULT_REWARD_FN = "examples/agent_loop/clone_reward.py:clone_accuracy_reward"


def _maybe_select(config: OmegaConf, path: str, default: Any) -> Any:
    value = OmegaConf.select(config, path)
    return default if value is None else value


def load_config(config_path: str, config_name: str, overrides: list[str]) -> OmegaConf:
    if not os.path.isdir(config_path):
        raise FileNotFoundError(f"Config directory not found: {config_path}")
    if config_name.endswith(".yaml"):
        config_name = config_name[: -len(".yaml")]
    with initialize_config_dir(config_dir=config_path):
        return compose(config_name=config_name, overrides=overrides)


def _expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def resolve_dataset_paths(dataset_paths: Iterable[str]) -> list[str]:
    resolved: list[str] = []
    for path in dataset_paths:
        expanded = _expand_path(path)
        if os.path.isdir(expanded):
            test_path = os.path.join(expanded, "test.parquet")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Expected test parquet at {test_path}")
            resolved.append(test_path)
        else:
            if not os.path.exists(expanded):
                raise FileNotFoundError(f"Dataset file not found: {expanded}")
            resolved.append(expanded)
    return resolved


def load_dataset(paths: list[str]) -> datasets.Dataset:
    return datasets.load_dataset("parquet", data_files=paths)["train"]


def sample_dataset(dataset: datasets.Dataset, max_samples: int, seed: int | None) -> datasets.Dataset:
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    if seed is None:
        return dataset.select(range(max_samples))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:max_samples].tolist()
    return dataset.select(indices)


async def run_samples(
    agent_loop: ToolAgentLoop,
    tokenizer,
    dataset: datasets.Dataset,
    sampling_params: dict[str, Any],
    *,
    print_responses: bool,
) -> None:
    total_reward = 0.0
    scored = 0
    exact = 0

    for idx, row in enumerate(dataset):
        raw_prompt = row["prompt"]
        reward_model = row.get("reward_model")
        data_source = row.get("data_source")
        extra_info = dict(row.get("extra_info") or {})
        extra_info.setdefault("interaction_kwargs", {})
        tools_kwargs = extra_info.get("tools_kwargs", {})
        sample_index = extra_info.get("index", idx)

        outputs = await agent_loop.run(
            sampling_params,
            raw_prompt=raw_prompt,
            multi_modal_data={},
            tools_kwargs=tools_kwargs,
            reward_model=reward_model,
            data_source=data_source,
            index=sample_index,
            extra_info=extra_info,
        )

        outputs_list = outputs if isinstance(outputs, list) else [outputs]
        root = outputs_list[0]
        reward = root.reward_score if root.reward_score is not None else 0.0
        total_reward += reward
        scored += 1
        if reward >= 1.0:
            exact += 1

        if print_responses:
            print(f"[{idx}] reward={reward}")
            for out in outputs_list:
                role = out.extra_fields.get("actor_role", "unknown")
                label = out.extra_fields.get("clone_label")
                text = tokenizer.decode(out.response_ids, skip_special_tokens=True)
                print(f"- role={role} label={label}")
                print(text)
                print("-" * 20)

    if scored == 0:
        print("No samples processed.")
        return

    mean_reward = total_reward / scored
    print(f"Processed {scored} samples | mean_reward={mean_reward:.4f} | exact={exact}/{scored}")


async def main(args):
    overrides = list(args.override or [])
    if args.model:
        overrides.append(f"actor_rollout_ref.model.path={args.model}")
    if args.dataset:
        overrides.append(f"data.val_files={args.dataset}")
    if args.reward_function:
        overrides.append(f"actor_rollout_ref.rollout.multi_turn.clone.reward_function={args.reward_function}")
    if args.tool_format:
        overrides.append(f"actor_rollout_ref.rollout.multi_turn.format={args.tool_format}")
    if args.prompt_length is not None:
        overrides.append(f"actor_rollout_ref.rollout.prompt_length={args.prompt_length}")
    if args.response_length is not None:
        overrides.append(f"actor_rollout_ref.rollout.response_length={args.response_length}")
    if args.gpu_memory_utilization is not None:
        overrides.append(f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_utilization}")
    if args.max_clones is not None:
        overrides.append(f"actor_rollout_ref.rollout.multi_turn.clone.max_clones_per_call={args.max_clones}")
    if args.max_clone_depth is not None:
        overrides.append(f"actor_rollout_ref.rollout.multi_turn.clone.max_clone_depth={args.max_clone_depth}")

    if torch.cuda.device_count() == 0:
        raise RuntimeError("vLLM rollout requires CUDA GPUs; none detected.")
    tensor_parallel_size = args.tensor_parallel_size or torch.cuda.device_count()
    overrides.append(f"actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_parallel_size}")

    config = load_config(args.config_path, args.config_name, overrides)

    reward_function = _maybe_select(
        config, "actor_rollout_ref.rollout.multi_turn.clone.reward_function", DEFAULT_REWARD_FN
    )
    if not reward_function:
        config.actor_rollout_ref.rollout.multi_turn.clone.reward_function = DEFAULT_REWARD_FN

    val_files = _maybe_select(config, "data.val_files", None)
    if val_files is None:
        raise ValueError("Validation dataset path is missing; set data.val_files or pass --dataset.")
    if isinstance(val_files, (list, tuple)):
        dataset_paths = [str(p) for p in val_files]
    else:
        dataset_paths = [str(val_files)]
    dataset_paths = resolve_dataset_paths(dataset_paths)
    dataset = load_dataset(dataset_paths)
    dataset = sample_dataset(dataset, args.max_samples, args.seed)

    ray.init(ignore_reinit_error=True)
    rollout_server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)
    rollout_server = rollout_server_class(
        replica_rank=0,
        config=config.actor_rollout_ref.rollout,
        model_config=config.actor_rollout_ref.model,
        gpus_per_node=config.actor_rollout_ref.rollout.tensor_model_parallel_size,
    )
    await rollout_server.init_standalone()

    server_manager = AsyncLLMServerManager(config=config, server_handles=[rollout_server.server_handle])

    model_path = _maybe_select(config, "actor_rollout_ref.model.path", None)
    if model_path is None:
        raise ValueError("Model path is missing; set actor_rollout_ref.model.path or pass --model.")
    trust_remote_code = bool(_maybe_select(config, "data.trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    agent_loop = ToolAgentLoop(
        trainer_config=_DummyConfig(config=config),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
    )

    sampling_params = {
        "temperature": args.temperature or config.actor_rollout_ref.rollout.temperature,
        "top_p": args.top_p or config.actor_rollout_ref.rollout.top_p,
        "max_new_tokens": args.max_new_tokens or config.actor_rollout_ref.rollout.response_length,
        "logprobs": False,
    }

    try:
        await run_samples(
            agent_loop,
            tokenizer,
            dataset,
            sampling_params,
            print_responses=args.print_responses,
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="examples/agent_loop/configs")
    parser.add_argument("--config-name", type=str, default="clone_tool_ppo")
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        help="Additional Hydra override strings (repeatable).",
    )
    parser.add_argument("--model", type=str, default=None, help="Override actor_rollout_ref.model.path.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override data.val_files (path or directory with test.parquet).",
    )
    parser.add_argument(
        "--reward-function",
        type=str,
        default=None,
        help=f"Override reward function path. Default: {DEFAULT_REWARD_FN}",
    )
    parser.add_argument("--tool-format", type=str, default=None)
    parser.add_argument("--prompt-length", type=int, default=None)
    parser.add_argument("--response-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-clones", type=int, default=None)
    parser.add_argument("--max-clone-depth", type=int, default=None)
    parser.add_argument("--print-responses", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(args))
