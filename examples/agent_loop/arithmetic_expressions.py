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
Preprocess locally generated arithmetic expressions into parquet format.

Input format (one per line):
    <expression>, <value>

Example:
    278 * 5345530 * 119 - 5684 * (7661185 - 5715) + 3041694 * (7772075 + 2326697), 30850701331748
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Tuple

import datasets

from verl.utils.hdfs_io import copy, makedirs


DEFAULT_SYSTEM_PROMPT = (
    "You are the root agent. Delegate the arithmetic by calling the tool `spawn_clone` as many times as needed. "
    "Coordinate the steps until the final value is computed. "
    "Put the final answer in the format `#### <value>`."
)


def parse_test_size(raw: str) -> float | int:
    if raw.isdigit():
        return int(raw)
    return float(raw)


def parse_expression_line(line: str, line_num: int) -> Tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "," not in stripped:
        raise ValueError(f"Line {line_num} is missing a comma separator: {stripped!r}")
    expression, value = stripped.rsplit(",", 1)
    expression = expression.strip()
    value = value.strip()
    if not expression:
        raise ValueError(f"Line {line_num} has an empty expression: {stripped!r}")
    if not value:
        raise ValueError(f"Line {line_num} has an empty value: {stripped!r}")
    return expression, value


def load_samples(input_path: str | None) -> list[Tuple[str, str]]:
    if input_path in (None, "-"):
        handle = sys.stdin
        should_close = False
    else:
        handle = open(input_path, "r", encoding="utf-8")
        should_close = True

    samples: list[Tuple[str, str]] = []
    try:
        for line_num, line in enumerate(handle, start=1):
            parsed = parse_expression_line(line, line_num)
            if parsed is not None:
                samples.append(parsed)
    finally:
        if should_close:
            handle.close()

    if not samples:
        raise ValueError("No valid expression/value pairs were found in the input.")

    return samples


def make_map_fn(split: str, data_source: str, system_prompt: str):
    def process_fn(example, idx: int):
        expression = example.pop("expression")
        answer = example.pop("answer")
        question = f"Calculate the value of {expression}."
        return {
            "data_source": data_source,
            "agent_name": "tool_agent",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer, "data_source": data_source},
            "extra_info": {
                "split": split,
                "index": idx,
                "expression": expression,
                "answer": answer,
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default=None,
        help="Path to a text file with '<expression>, <value>' per line. Use '-' or omit to read stdin.",
    )
    parser.add_argument("--data_source", default="synthetic_arithmetic", help="Dataset identifier for reward scoring.")
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used for the root agent.",
    )
    parser.add_argument("--test_size", type=parse_test_size, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_save_dir",
        default="~/data/arithmetic_expressions",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()

    samples = load_samples(args.input_file)
    records = [{"expression": expr, "answer": answer} for expr, answer in samples]
    dataset = datasets.Dataset.from_list(records)

    if len(dataset) < 2 or args.test_size == 0:
        train_dataset = dataset
        test_dataset = dataset.select([])
    else:
        split = dataset.train_test_split(test_size=args.test_size, seed=args.seed, shuffle=True)
        train_dataset = split["train"]
        test_dataset = split["test"]

    train_dataset = train_dataset.map(
        function=make_map_fn("train", args.data_source, args.system_prompt),
        with_indices=True,
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test", args.data_source, args.system_prompt),
        with_indices=True,
    )

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
