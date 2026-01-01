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
"""
Drop-in reward function for ToolAgentLoop clone rollouts.

The agent loop now propagates dataset metadata (data_source, reward_model)
into `AgentLoopOutput.extra_fields`, so this function can fetch the ground-truth
answer that was stored in the dataset row.
"""

from __future__ import annotations

from typing import Any

from verl.utils.reward_score import default_compute_score


def _coerce_ground_truth(ground_truth: Any) -> Any:
    if isinstance(ground_truth, list) and len(ground_truth) == 1:
        return ground_truth[0]
    return ground_truth


def _normalize_root_answer_for_scoring(data_source: str | None, root_answer: str) -> str:
    if not root_answer:
        return root_answer
    if data_source == "openai/gsm8k" and "####" not in root_answer:
        # GSM8K's default scorer extracts answers from the "#### <num>" pattern.
        return f"#### {root_answer.strip()}"
    return root_answer


def clone_accuracy_reward(
    root_answer: str, clone_rollouts: list[Any], metadata: dict[str, Any]
) -> dict[str, Any] | float:
    """
    Compute a single scalar reward for the root + all clones.

    Args:
        root_answer: decoded root response text with chain-of-thought stripped.
        clone_rollouts: list of AgentLoopOutput objects returned by spawn_clone.
        metadata: dict passed by ToolAgentLoop._assign_rewards. Contains:
            - root_extra: extra_fields from the root rollout (now includes
              reward_model/data_source/extra_info from the dataset row).
            - num_clones: number of clone rollouts.

    Returns:
        Either a float or a dict with a `reward` key; any extra keys will be
        recorded in reward_extra_info.
    """

    root_extra = metadata.get("root_extra") or {}
    reward_model = root_extra.get("reward_model") or {}
    data_source = root_extra.get("data_source") or reward_model.get("data_source")
    ground_truth = _coerce_ground_truth(reward_model.get("ground_truth"))

    if ground_truth is None:
        return {"reward": 0.0, "reason": "missing_ground_truth"}

    try:
        score = 1.0 if int(root_answer.strip()) == int(ground_truth.strip()) else 0.1
    except Exception:
        # Fallback: exact/substring match
        score = 1.0 if str(ground_truth).strip() in root_answer else 0.0

    return {
        "reward": score,
        "score": score,
        "num_clones": metadata.get("num_clones", len(clone_rollouts)),
        "ground_truth": ground_truth,
    }


__all__ = ["clone_accuracy_reward"]
