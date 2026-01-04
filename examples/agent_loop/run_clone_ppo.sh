#!/usr/bin/env bash
set -euo pipefail

# Minimal entrypoint to launch PPO training with the ToolAgentLoop clone tool.
# Fill in MODEL_PATH / DATA_PATHS or override via environment variables.

MODEL_PATH=${MODEL_PATH:-/path/to/model}
TRAIN_PATH=${TRAIN_PATH:-/path/to/train.parquet}
VAL_PATH=${VAL_PATH:-/path/to/val.parquet}

python -m verl.trainer.main_ppo \
  --config-path examples/agent_loop/configs \
  --config-name clone_tool_ppo \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  data.train_files="${TRAIN_PATH}" \
  data.val_files="${VAL_PATH}" \
  trainer.project_name=clone_tool_agent \
  trainer.experiment_name=first_pass \
  "$@"
