# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Concurrent variant of NaiveRewardManager.

Differences vs `naive`:
  - The per-sample `compute_score` calls in the batch are dispatched in
    parallel via a ThreadPoolExecutor. This is required when each
    `compute_score` itself issues a (slow) LLM API call (as in our
    set_selection / RubricRanker reward), so that one rollout batch only
    pays roughly one LLM round-trip, instead of len(batch) round-trips.
  - All other behavior (decoding, tensor layout, debug printing,
    reward_extra_info handling) is kept identical to NaiveRewardManager.
"""

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive_concurrent")
class NaiveConcurrentRewardManager(AbstractRewardManager):
    """Same behavior as NaiveRewardManager, but per-sample compute_score
    calls run concurrently in a thread pool."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_workers: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        # Default to a generous pool size since the work is I/O-bound (HTTP
        # to the GPT eval API). Override via env var if needed.
        self.max_workers = max_workers or int(
            os.getenv("REWARD_MANAGER_MAX_WORKERS", "128")
        )

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # If there is rm score, we directly return rm score. Otherwise, we
        # compute via rm_score_fn.
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # ----- Stage 1: assemble per-sample inputs (sequential, fast) -----
        per_sample = []  # list of dicts, one per sample, in batch order
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            per_sample.append({
                "valid_response_length": int(valid_response_length),
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "extra_info": extra_info,
            })

        # ----- Stage 2: dispatch compute_score concurrently -----
        def _run(idx):
            s = per_sample[idx]
            return idx, self.compute_score(
                data_source=s["data_source"],
                solution_str=s["response_str"],
                ground_truth=s["ground_truth"],
                extra_info=s["extra_info"],
            )

        scores = [None] * len(per_sample)
        n_workers = max(1, min(self.max_workers, len(per_sample)))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for idx, score in ex.map(_run, range(len(per_sample))):
                scores[idx] = score

        # ----- Stage 3: write results into the reward tensor + debug prints -----
        for i, s in enumerate(per_sample):
            score = scores[i]
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, s["valid_response_length"] - 1] = reward

            data_source = s["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", s["prompt_str"])
                print("[response]", s["response_str"])
                print("[ground_truth]", s["ground_truth"])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
