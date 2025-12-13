# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import copy
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, _DummyConfig, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import add_generation_prompt_for_gpt_oss, format_gpt_oss_tool_response_manually
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


@dataclass
class CloneRunResult:
    """Container for clone execution artifacts."""

    clone_id: str
    objective: str
    label: str
    visible_text: str
    rollouts: list[AgentLoopOutput]


@dataclass
class ToolInvocationResult:
    response: ToolResponse
    reward: float | None
    extra: dict[str, Any]


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # Extra fields for dynamic addition
        self.extra_fields: dict[str, Any] = {}

        # Clone orchestration
        self.clone_rollouts: list[AgentLoopOutput] = []


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.clone_config = cls._load_clone_runtime_config(config)
        cls.clone_tool_name = cls.clone_config["tool_name"]
        if cls.clone_config["enabled"]:
            cls.clone_tool_schema = cls._build_clone_tool_schema(cls.clone_config)
            cls.tool_schemas.append(cls.clone_tool_schema.model_dump(exclude_unset=True, exclude_none=True))
        cls.clone_reward_fn = cls._load_reward_fn(cls.clone_config)
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format
        print(f"Initialized tools: {cls.tools}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = initialize_system_prompt(cls.tokenizer, **cls.apply_chat_template_kwargs)

        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput | list[AgentLoopOutput]:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        agent_data.extra_fields["clone_depth"] = kwargs.get("clone_depth", 0)
        agent_data.extra_fields["parent_request_id"] = kwargs.get("parent_request_id")
        agent_data.extra_fields["clone_label"] = kwargs.get("clone_label")
        agent_data.extra_fields["clone_allow_tools"] = kwargs.get("clone_allow_tools", True)

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data, sampling_params)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        output.extra_fields.update(
            {
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "clone_depth": agent_data.extra_fields.get("clone_depth", 0),
                "parent_request_id": agent_data.extra_fields.get("parent_request_id"),
                "actor_role": "root" if agent_data.extra_fields.get("clone_depth", 0) == 0 else "clone",
                "request_id": request_id,
            }
        )
        if agent_data.clone_rollouts:
            output.extra_fields["clone_children"] = [
                clone.extra_fields.get("clone_id") for clone in agent_data.clone_rollouts
            ]
            output.extra_fields["clone_summaries"] = agent_data.extra_fields.get("clone_summaries", [])
            # ensure turn level rewards persist across rollouts
            for clone in agent_data.clone_rollouts:
                clone.extra_fields.setdefault("turn_scores", [])
                clone.extra_fields.setdefault("tool_rewards", [])

        # Assign a single reward to root + clones to bypass default reward loop
        reward_value = self._assign_rewards(output, agent_data.clone_rollouts)
        output.extra_fields["reward_source"] = "clone_reward_fn"
        outputs = [output, *agent_data.clone_rollouts] if agent_data.clone_rollouts else [output]
        for rollout in outputs:
            rollout.reward_score = reward_value
            rollout.extra_fields.setdefault("reward_source", "clone_reward_fn")
        return outputs if len(outputs) > 1 else outputs[0]

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        tool_schemas = self._active_tool_schemas(agent_data)
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    tools=tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data, sampling_params))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses: list[ToolInvocationResult] = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        for invocation_result in responses:
            tool_response = invocation_result.response
            tool_reward = invocation_result.reward
            extra = invocation_result.extra or {}

            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:
                            new_images_this_turn.append(img)
                else:
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)

            # Handle video data
            if tool_response.video:
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

            if extra.get("clone_rollouts"):
                agent_data.clone_rollouts.extend(extra["clone_rollouts"])
                agent_data.extra_fields.setdefault("clone_summaries", []).extend(extra.get("clone_summaries", []))

        agent_data.messages.extend(add_messages)
        # Update prompt with tool responses
        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            current_images = new_images_this_turn if new_images_this_turn else None
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if self.tool_parser_name == "gpt-oss":
                logger.info("manually format tool responses for gpt-oss")
                tool_response_texts = []
                for i, tool_msg in enumerate(add_messages):
                    actual_tool_name = tool_call_names[i]
                    formatted = format_gpt_oss_tool_response_manually(tool_msg["content"], actual_tool_name)
                    tool_response_texts.append(formatted)

                tool_response_text = add_generation_prompt_for_gpt_oss("".join(tool_response_texts))
                response_ids = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
                )
            else:
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
                )
                response_ids = response_ids[len(self.system_prompt) :]
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        if self.processor is not None:
            raw_user_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt) :]

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> ToolInvocationResult:
        """Call tool and return tool response."""
        tool_name = tool_call.name
        try:
            tool_args = json.loads(tool_call.arguments)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Error decoding tool arguments for {tool_name}: {e}")
            return ToolInvocationResult(
                response=ToolResponse(text=f"Error when decoding tool arguments for {tool_name}: {e}"),
                reward=0.0,
                extra={},
            )

        # Handle cloning as a built-in tool
        if self.clone_config["enabled"] and tool_name == self.clone_tool_name:
            response, reward, extra = await self._execute_clone_tool(tool_args, agent_data, sampling_params)
            return ToolInvocationResult(response=response, reward=reward, extra=extra)

        tool, instance_id = None, None
        try:
            if tool_name not in self.tools:
                raise KeyError(f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}")

            tool = self.tools[tool_name]
            kwargs = agent_data.tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.warning(f"Error when executing tool '{tool_name}': {e}")
            return ToolInvocationResult(
                response=ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                reward=0.0,
                extra={},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolInvocationResult(
            response=ToolResponse(**tool_response_kwargs), reward=tool_reward, extra=res or {}
        )

    def _active_tool_schemas(self, agent_data: AgentData) -> list[dict[str, Any]]:
        """Return the tool schemas that should be exposed for this request."""
        if not agent_data.extra_fields.get("clone_allow_tools", True):
            return []
        if not self.clone_config.get("enabled", False):
            return self.tool_schemas

        depth = agent_data.extra_fields.get("clone_depth", 0)
        if depth >= self.clone_config["max_clone_depth"]:
            return [
                schema
                for schema in self.tool_schemas
                if schema.get("function", {}).get("name") != self.clone_tool_name
            ]
        return self.tool_schemas

    @classmethod
    def _load_clone_runtime_config(cls, config):
        clone_cfg = getattr(config.actor_rollout_ref.rollout.multi_turn, "clone", None)

        def _get(key, default):
            if clone_cfg is None:
                return default
            if isinstance(clone_cfg, dict):
                return clone_cfg.get(key, default)
            if hasattr(clone_cfg, key):
                return getattr(clone_cfg, key)
            try:
                return clone_cfg[key]
            except Exception:
                return default

        final_markers = _get(
            "final_answer_markers",
            ["</think>", "Final Answer:", "final answer:", "Answer:", "Result:", "OUTPUT:"],
        )
        if isinstance(final_markers, str):
            final_markers = [final_markers]

        return {
            "enabled": _get("enable", True),
            "tool_name": _get("tool_name", "spawn_clone"),
            "max_clones_per_call": _get("max_clones_per_call", 3),
            "max_clone_depth": _get("max_clone_depth", 1),
            "allow_tool_use": _get("allow_tool_use", True),
            "allow_nested_clones": _get("allow_nested_clones", False),
            "system_prompt": _get(
                "system_prompt",
                (
                    "You are a helper clone spawned by a root agent. "
                    "Use a <think>...</think> block for reasoning, then provide a concise final answer. "
                    "Prefer short, actionable responses and conserve tokens."
                ),
            ),
            "final_answer_markers": final_markers,
            "reward_function": _get("reward_function", None),
        }

    @classmethod
    def _build_clone_tool_schema(cls, clone_config: dict[str, Any]) -> OpenAIFunctionToolSchema:
        """Build a simple schema that advertises the clone tool to the root agent."""
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name=clone_config["tool_name"],
                description="Fork helper clones of the current model to work on subtasks and report back succinctly.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "objectives": OpenAIFunctionPropertySchema(
                            type="array",
                            description="List of objectives or messages for each clone (string or object).",
                        ),
                        "shared_context": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Minimal context to share with each clone to avoid re-tokenizing long history.",
                        ),
                        "allow_tools": OpenAIFunctionPropertySchema(
                            type="boolean",
                            description="Let clones use the same toolset; defaults to config.",
                        ),
                    },
                    required=["objectives"],
                ),
            ),
        )

    def _strip_thinking(self, text: str) -> str:
        """Remove chain-of-thought markers and return the final answer."""
        if not text:
            return ""

        stripped = text
        if "</think>" in stripped:
            stripped = stripped.split("</think>")[-1]

        for marker in self.clone_config["final_answer_markers"]:
            if marker in stripped:
                stripped = stripped.split(marker, 1)[-1]

        stripped = stripped.strip()
        if stripped:
            return stripped

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    @classmethod
    def _load_reward_fn(cls, clone_config: dict[str, Any]):
        """Load a user-provided reward function or fall back to a default no-op."""

        def _default_reward_fn(root_answer: str, clone_rollouts: list[AgentLoopOutput], metadata: dict[str, Any]):
            return 0.0

        path = clone_config.get("reward_function")
        if not path:
            return _default_reward_fn

        module_path, _, func_name = path.replace(":", ".").rpartition(".")
        if not module_path or not func_name:
            logger.warning("Invalid reward_function path '%s'; using default reward fn", path)
            return _default_reward_fn

        try:
            module = __import__(module_path, fromlist=[func_name])
            reward_fn = getattr(module, func_name)
            return reward_fn
        except (ImportError, AttributeError) as exc:
            logger.warning("Failed to import reward_function '%s': %s; using default reward fn", path, exc)
            return _default_reward_fn

    def _assign_rewards(self, root_output: AgentLoopOutput, clone_rollouts: list[AgentLoopOutput]) -> float:
        """Assign a single reward to root and all clones, bypassing default reward loop."""
        try:
            decoded = self.tokenizer.decode(root_output.response_ids, skip_special_tokens=True)
            root_answer = self._strip_thinking(decoded)
        except Exception as exc:  # noqa: BLE001
            logger.error("[SHOULDN'T HAPPEN] Failed to decode root answer for reward fn: %s", exc)
            root_answer = ""

        try:
            reward_value = self.clone_reward_fn(
                root_answer,
                clone_rollouts or [],
                {
                    "root_extra": root_output.extra_fields,
                    "num_clones": len(clone_rollouts or []),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reward function raised; defaulting to 0.0: %s", exc)
            reward_value = 0.0

        if isinstance(reward_value, dict) and "reward" in reward_value:
            reward_value = reward_value["reward"]
        try:
            reward_value = float(reward_value)
        except Exception:
            reward_value = 0.0

        return reward_value

    def _build_clone_prompt(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        """Create a compact prompt for a clone instance."""
        allow_tools = request.get("allow_tools", self.clone_config["allow_tool_use"])
        user_sections = [request["instruction"]]
        if request.get("shared_context"):
            user_sections.append(f"Shared context from root:\n{request['shared_context']}")
        if not allow_tools:
            user_sections.append("Do not call any tools; respond directly.")
        else:
            user_sections.append("Use tools only if it shortens the path to the answer.")

        return [
            {"role": "system", "content": self.clone_config["system_prompt"]},
            {"role": "user", "content": "\n\n".join([section for section in user_sections if section])},
        ]

    def _normalize_clone_requests(self, tool_args: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize user-specified clone objectives into a list of requests."""
        if not isinstance(tool_args, dict):
            return []

        shared_context = tool_args.get("shared_context") or tool_args.get("context")
        allow_tools = tool_args.get("allow_tools")
        sampling_overrides = tool_args.get("sampling_overrides")
        if sampling_overrides is not None and not isinstance(sampling_overrides, dict):
            sampling_overrides = None

        raw_requests = tool_args.get("objectives") or tool_args.get("tasks")
        if raw_requests is None and "objective" in tool_args:
            raw_requests = [tool_args["objective"]]
        if raw_requests is None:
            return []
        if isinstance(raw_requests, str):
            raw_requests = [raw_requests]

        names = tool_args.get("names") if isinstance(tool_args.get("names"), list) else None
        normalized: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_requests):
            if isinstance(item, str):
                name = names[idx] if names and idx < len(names) else None
                normalized.append(
                    {
                        "instruction": item,
                        "shared_context": shared_context,
                        "name": name,
                        "allow_tools": allow_tools,
                        "sampling_overrides": sampling_overrides,
                    }
                )
            elif isinstance(item, dict):
                instruction = item.get("instruction") or item.get("objective") or item.get("task")
                if not instruction:
                    continue
                normalized.append(
                    {
                        "instruction": instruction,
                        "shared_context": item.get("shared_context", shared_context),
                        "name": item.get("name") or item.get("label"),
                        "allow_tools": item.get("allow_tools", allow_tools),
                        "sampling_overrides": item.get("sampling_overrides", sampling_overrides),
                    }
                )

        for request in normalized:
            if request.get("allow_tools") is None:
                request["allow_tools"] = self.clone_config["allow_tool_use"]
        return normalized

    async def _run_single_clone(
        self, request: dict[str, Any], agent_data: AgentData, sampling_params: dict[str, Any], index: int
    ) -> CloneRunResult:
        """Spin up a clone ToolAgentLoop using the same weights and context."""
        clone_id = uuid4().hex
        clone_depth = agent_data.extra_fields.get("clone_depth", 0) + 1
        clone_label = request.get("name") or f"clone_{index}"

        clone_sampling = dict(sampling_params)
        if request.get("sampling_overrides"):
            clone_sampling.update(request["sampling_overrides"])

        clone_prompt = self._build_clone_prompt(request)
        clone_loop = self.__class__(
            trainer_config=_DummyConfig(config=self.config),
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        clone_kwargs = {
            "raw_prompt": clone_prompt,
            "multi_modal_data": {},
            "tools_kwargs": agent_data.tools_kwargs,
            "extra_info": {"interaction_kwargs": {}},
            "clone_depth": clone_depth,
            "parent_request_id": agent_data.request_id,
            "clone_label": clone_label,
            "clone_allow_tools": request.get("allow_tools", self.clone_config["allow_tool_use"]),
        }

        clone_outputs = await clone_loop.run(clone_sampling, **clone_kwargs)
        clone_outputs_list = clone_outputs if isinstance(clone_outputs, list) else [clone_outputs]
        for output in clone_outputs_list:
            output.extra_fields.setdefault("actor_role", "clone")
            output.extra_fields.setdefault("clone_depth", clone_depth)
            output.extra_fields.setdefault("parent_request_id", agent_data.request_id)
            output.extra_fields.setdefault("clone_id", clone_id)
            output.extra_fields.setdefault("clone_label", clone_label)
            output.extra_fields.setdefault("clone_objective", request["instruction"])
            output.extra_fields.setdefault("reward_extra_info", {})
            output.extra_fields.setdefault("turn_scores", [])
            output.extra_fields.setdefault("tool_rewards", [])
            output.extra_fields["raw_prompt_override"] = clone_prompt

        main_output = clone_outputs_list[0]
        raw_text = self.tokenizer.decode(main_output.response_ids, skip_special_tokens=True)
        visible_text = self._strip_thinking(raw_text)

        return CloneRunResult(
            clone_id=clone_id,
            objective=request["instruction"],
            label=clone_label,
            visible_text=visible_text,
            rollouts=clone_outputs_list,
        )

    async def _execute_clone_tool(
        self, tool_args: dict[str, Any], agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> tuple[ToolResponse, float | None, dict[str, Any]]:
        """Execute the built-in clone tool."""
        current_depth = agent_data.extra_fields.get("clone_depth", 0)
        if current_depth >= self.clone_config["max_clone_depth"]:
            return (
                ToolResponse(text="Clone tool disabled because maximum clone depth has been reached."),
                None,
                {},
            )

        requests = self._normalize_clone_requests(tool_args)
        if not requests:
            return ToolResponse(text="No clone objectives provided."), None, {}

        clone_tool_overrides = agent_data.tools_kwargs.get(self.clone_tool_name, {})
        max_clones = clone_tool_overrides.get("max_clones_per_call", self.clone_config["max_clones_per_call"])
        try:
            max_clones = int(max_clones)
        except Exception:
            max_clones = self.clone_config["max_clones_per_call"]

        if len(requests) > max_clones:
            logger.info(
                "Truncating clone requests from %d to max_clones_per_call=%d",
                len(requests),
                max_clones,
            )
            requests = requests[:max_clones]

        tasks = [
            self._run_single_clone(request, agent_data, sampling_params, index=i) for i, request in enumerate(requests)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        clone_rollouts: list[AgentLoopOutput] = []
        clone_summaries: list[dict[str, Any]] = []
        errors: list[str] = []
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
                continue
            clone_rollouts.extend(result.rollouts)
            clone_summaries.append(
                {
                    "clone_id": result.clone_id,
                    "label": result.label,
                    "objective": result.objective,
                    "answer": result.visible_text,
                }
            )

        lines = [f"{summary['label']}: {summary['answer']}" for summary in clone_summaries]
        if errors:
            lines.append(f"Errors while running some clones: {errors}")
        tool_response_text = "\n".join(lines) if lines else "No clone returned a response."

        extra = {
            "clone_rollouts": clone_rollouts,
            "clone_summaries": clone_summaries,
            "clone_depth": current_depth,
        }
        return ToolResponse(text=tool_response_text), None, extra

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map
