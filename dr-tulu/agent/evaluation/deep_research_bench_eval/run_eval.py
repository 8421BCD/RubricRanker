#!/usr/bin/env python3
"""
Self-contained Deep Research Bench (DRB) evaluation script.

Runs both RACE (article quality) and FACT (citation verification) evaluations
without requiring the external deep_research_bench repository.

Usage:
    # Full pipeline: format + RACE + FACT
    python run_eval.py --input_file <path/to/drb_output.jsonl> --task_name <name>

    # Only RACE evaluation (skip FACT)
    python run_eval.py --input_file <path/to/drb_output.jsonl> --task_name <name> --skip_fact

    # Only FACT evaluation (skip RACE)
    python run_eval.py --input_file <path/to/drb_output.jsonl> --task_name <name> --skip_race

    # With options
    python run_eval.py --input_file <path/to/drb_output.jsonl> --task_name <name> \
        --max_workers 10 --only_en --limit 5

Environment Variables Required:
    GEMINI_API_KEY: Google Gemini API key (for both RACE and FACT evaluation)

Data files (bundled in data/ subdirectory):
    - data/prompt_data/query.jsonl
    - data/criteria_data/criteria.jsonl
    - data/test_data/cleaned_data/reference.jsonl
"""

import argparse
import concurrent.futures
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================================
# Logging setup
# ============================================================================
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================================
# Constants
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
HF_EVAL_DATA_REPO = "rl-research/dr-tulu-eval-data"
MAX_RETRIES = 10

# Gemini model configuration
FACT_MODEL = "gemini-2.5-flash"
RACE_MODEL = "gemini-2.5-pro"


def _ensure_data_files() -> tuple:
    """
    Ensure DRB evaluation data files are available locally.
    Downloads from HuggingFace if not found in local data/ directory.

    Returns:
        (criteria_file, reference_file, query_file) paths
    """
    criteria_file = DATA_DIR / "criteria_data" / "criteria.jsonl"
    reference_file = DATA_DIR / "test_data" / "cleaned_data" / "reference.jsonl"
    query_file = DATA_DIR / "prompt_data" / "query.jsonl"

    # Check if all files exist locally
    if criteria_file.exists() and reference_file.exists() and query_file.exists():
        return criteria_file, reference_file, query_file

    # Download from HuggingFace
    logger.info(f"Downloading DRB evaluation data from {HF_EVAL_DATA_REPO}...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub required to download eval data. "
            "Install with: pip install huggingface_hub"
        )

    cache_dir = DATA_DIR / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    hf_files = {
        "deep_research_bench/query.jsonl": query_file,
        "deep_research_bench/criteria.jsonl": criteria_file,
        "deep_research_bench/reference.jsonl": reference_file,
    }
    for hf_path, local_path in hf_files.items():
        if not local_path.exists():
            logger.info(f"  Downloading {hf_path}...")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=HF_EVAL_DATA_REPO,
                filename=hf_path,
                repo_type="dataset",
                cache_dir=str(cache_dir),
            )
            # Symlink or copy to expected path
            import shutil
            shutil.copy2(downloaded, local_path)

    logger.info("DRB evaluation data ready.")
    return criteria_file, reference_file, query_file


# Resolved at runtime
CRITERIA_FILE = None
REFERENCE_FILE = None
QUERY_FILE = None


# ============================================================================
# Utility functions
# ============================================================================
def load_jsonl(file_path: str) -> list:
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def extract_json_from_markdown(text: str) -> Optional[str]:
    """Extract JSON from a markdown text that may contain ```json ... ``` blocks."""
    if not isinstance(text, str):
        return None

    # Method 0: Direct JSON object
    if text.strip().startswith("{") and text.strip().endswith("}"):
        try:
            json.loads(text.strip())
            return text.strip()
        except json.JSONDecodeError:
            pass

    # Method 1: Code block extraction
    if "```json" in text and "```" in text[text.find("```json") + 7 :]:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            json_str = text[start:end].strip()
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass

    # Method 2: Regex matching
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        json_str = match.group(1).strip()
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass

    # Method 3: Full text as JSON
    try:
        json.loads(text.strip())
        return text.strip()
    except json.JSONDecodeError:
        pass

    # Method 4: Outermost curly braces
    start = text.find("{")
    if start != -1:
        level = 0
        for i, char in enumerate(text[start:]):
            if char == "{":
                level += 1
            elif char == "}":
                level -= 1
                if level == 0:
                    end = start + i + 1
                    potential_json = text[start:end]
                    try:
                        json.loads(potential_json)
                        return potential_json
                    except json.JSONDecodeError:
                        pass
                    break

    # Method 5: First/last brace matching
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        potential_json = text[start : end + 1]
        try:
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            pass

    # Method 6: Keyword pattern matching fallback
    if (
        "comprehensiveness" in text
        and "article_1_score" in text
        and "article_2_score" in text
    ):
        try:
            dimensions = [
                "comprehensiveness",
                "insight",
                "instruction_following",
                "readability",
            ]
            result = {}
            for dim in dimensions:
                if dim in text:
                    result[dim] = []
                    dim_start = text.find(f'"{dim}"')
                    if dim_start == -1:
                        dim_start = text.find(f"'{dim}'")
                    if dim_start == -1:
                        dim_start = text.find(dim)
                    if dim_start != -1:
                        next_dim_start = len(text)
                        for next_dim in dimensions:
                            if next_dim != dim:
                                pos = text.find(f'"{next_dim}"', dim_start)
                                if pos == -1:
                                    pos = text.find(f"'{next_dim}'", dim_start)
                                if pos == -1:
                                    pos = text.find(next_dim, dim_start + len(dim))
                                if pos != -1 and pos < next_dim_start:
                                    next_dim_start = pos
                        dim_content = text[dim_start:next_dim_start]
                        criterion_matches = re.finditer(
                            r'"criterion"\s*:\s*"([^"]+)"', dim_content
                        )
                        score1_matches = re.finditer(
                            r'"article_1_score"\s*:\s*(\d+\.?\d*)', dim_content
                        )
                        score2_matches = re.finditer(
                            r'"article_2_score"\s*:\s*(\d+\.?\d*)', dim_content
                        )
                        criteria = [m.group(1) for m in criterion_matches]
                        scores1 = [float(m.group(1)) for m in score1_matches]
                        scores2 = [float(m.group(1)) for m in score2_matches]
                        for i in range(min(len(criteria), len(scores1), len(scores2))):
                            result[dim].append(
                                {
                                    "criterion": criteria[i],
                                    "article_1_score": scores1[i],
                                    "article_2_score": scores2[i],
                                }
                            )
            if any(len(scores) > 0 for scores in result.values()):
                return json.dumps(result)
        except Exception:
            pass

    return None


def contains_chinese(text: str) -> bool:
    """Check if a string contains Chinese characters."""
    return re.search(r"[\u4e00-\u9fff]", text) is not None


# ============================================================================
# Gemini API Client
# ============================================================================
class GeminiClient:
    """Client for calling Gemini API."""

    def __init__(self, api_key: str = None, model: str = RACE_MODEL):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai package required. Install with: pip install google-genai"
            )

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY environment variable."
            )

        self.client = genai.Client(
            api_key=self.api_key, http_options={"timeout": 600000}
        )
        self.model = model
        self._genai = genai
        self._types = types

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: str = None,
    ) -> str:
        """Generate text response from Gemini."""
        model_to_use = model or self.model
        contents = []
        if system_prompt:
            contents.append({"role": "system", "parts": [{"text": system_prompt}]})
        contents.append({"role": "user", "parts": [{"text": user_prompt}]})

        response = self.client.models.generate_content(
            model=model_to_use,
            contents=contents,
            config=self._types.GenerateContentConfig(
                thinking_config=self._types.ThinkingConfig(thinking_budget=16000)
            ),
        )
        return response.text


# ============================================================================
# Scoring prompts (English)
# ============================================================================
SCORE_PROMPT_EN = """
<system_role>You are a strict, meticulous, and objective research article evaluation expert. You excel at using specific assessment criteria to deeply compare two articles on the same task, providing precise scores and clear justifications.</system_role>

<user_prompt>
**Task Background**
There is a deep research task, and you need to evaluate two research articles written for this task. We will assess the articles across four dimensions: Comprehensiveness, Insight, Instruction Following, and Readability. The content is as follows:
<task>
"{task_prompt}"
</task>

**Articles to Evaluate**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**Evaluation Criteria**
Now, you need to evaluate and compare these two articles based on the following **evaluation criteria list**, providing comparative analysis and scoring each on a scale of 0-10. Each criterion includes an explanation, please understand carefully.

<criteria_list>
{criteria_list}
</criteria_list>

<Instruction>
**Your Task**
Please strictly evaluate and compare `<article_1>` and `<article_2>` based on **each criterion** in the `<criteria_list>`. You need to:
1.  **Analyze Each Criterion**: Consider how each article fulfills the requirements of each criterion.
2.  **Comparative Evaluation**: Analyze how the two articles perform on each criterion, referencing the content and criterion explanation.
3.  **Score Separately**: Based on your comparative analysis, score each article on each criterion (0-10 points).

**Scoring Rules**
For each criterion, score both articles on a scale of 0-10 (continuous values). The score should reflect the quality of performance on that criterion:
*   0-2 points: Very poor performance.
*   2-4 points: Poor performance.
*   4-6 points: Average performance.
*   6-8 points: Good performance.
*   8-10 points: Excellent/outstanding performance.

**Output Format Requirements**
Please **strictly** follow the `<output_format>` below for each criterion evaluation.
</Instruction>

<output_format>
{{
    "comprehensiveness": [
        {{
            "criterion": [Text content of the first comprehensiveness evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        }},
        ...
    ],
    "insight": [...],
    "instruction_following": [...],
    "readability": [...]
}}
</output_format>

Now, please evaluate the two articles. Ensure your output follows the specified format and that the JSON format is parsable.
</user_prompt>
"""

# ============================================================================
# Scoring prompts (Chinese)
# ============================================================================
SCORE_PROMPT_ZH = """
<system_role>你是一名严格、细致、客观的调研文章评估专家。你擅长根据具体的评估标准，深入比较两篇针对同一任务的文章，并给出精确的评分和清晰的理由。</system_role>

<user_prompt>
**任务背景**
有一个深度调研任务，你需要评估针对该任务撰写的两篇调研文章。我们会从以下四个维度评估文章：全面性、洞察力、指令遵循能力和可读性。内容如下：
<task>
"{task_prompt}"
</task>

**待评估文章**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**评估标准**
现在，你需要根据以下**评判标准列表**，逐条评估并比较这两篇文章的表现，输出对比分析，然后给出0-10的分数。

<criteria_list>
{criteria_list}
</criteria_list>

<Instruction>
**你的任务**
请严格按照 `<criteria_list>` 中的**每一条标准**，对比评估 `<article_1>` 和 `<article_2>`。
1. 逐条分析
2. 对比评估
3. 分别打分（0-10分）

**输出格式要求**
请**严格**按照下列格式输出：
</Instruction>

<output_format>
{{
    "comprehensiveness": [
        {{
            "criterion": [评判标准文本],
            "analysis": [对比分析],
            "article_1_score": [0-10连续分数],
            "article_2_score": [0-10连续分数]
        }},
        ...
    ],
    "insight": [...],
    "instruction_following": [...],
    "readability": [...]
}}
</output_format>

请确保输出格式遵守上述要求，且JSON格式可解析。
</user_prompt>
"""

# ============================================================================
# Article cleaning prompts
# ============================================================================
CLEAN_PROMPT_EN = """
<system_role>You are a professional article editor who is good at cleaning and refining article content.</system_role>

<user_prompt>
Please help me clean the following research article, removing all citation links, citation marks (such as [1], [2], 1, 2, etc. or other complex citation formats), reference lists, footnotes, and ensuring the content is coherent and smooth.
Keep all other original content of the article, removing only the citations. If the content of the citation mark is used as part of a sentence in the article, keep the text content and remove other marks.

Article content:
"{article}"

Please return the cleaned article in full, without adding any additional comments or explanations.
</user_prompt>
"""

CLEAN_PROMPT_ZH = """
<system_role>你是一名专业的文章编辑，擅长整理和清洗文章内容。</system_role>

<user_prompt>
请帮我清洗以下研究文章，去除所有引用链接、引用标记（如[1]、[2]、1、2 等或其他复杂引用格式）、参考文献列表、脚注，并确保文章内容连贯流畅。
保留文章的所有其他原本内容、只移除引用。如果文章中使用引用标记中的内容作为语句的一部分，保留这其中的文字内容，移除其他标记。

文章内容：
"{article}"

请返回清洗后的文章全文，不要添加任何额外说明或评论。
</user_prompt>
"""

# ============================================================================
# FACT validation prompts
# ============================================================================
FACT_VALIDATE_PROMPT_EN = """You will be provided with a reference and some statements. Please determine whether each statement is 'supported', 'unsupported', or 'unknown' with respect to the reference. Please note:
First, assess whether the reference contains any valid content. If the reference contains no valid information, such as a 'page not found' message, then all statements should be considered 'unknown'.
If the reference is valid, for a given statement: if the facts or data it contains can be found entirely or partially within the reference, it is considered 'supported' (data accepts rounding); if all facts and data in the statement cannot be found in the reference, it is considered 'unsupported'.

You should return the result in a JSON list format, where each item in the list contains the statement's index and the judgment result, for example:
[
    {{
        "idx": 1,
        "result": "supported"
    }},
    {{
        "idx": 2,
        "result": "unsupported"
    }}
]

Below are the reference and statements:
<reference>
{reference}
</reference>

<statements>
{statements}
</statements>

Begin the assessment now. Output only the JSON list, without any conversational text or explanations."""

FACT_VALIDATE_PROMPT_ZH = """你会看到一个参考资料和一些statement，请你判断对于参考资料来说statement是supported、unsupported、或者unknown，注意：
首先判断参考资料是否存在有效内容，如果参考资料中没有任何有效信息，如"page not found"页面，则认为所有statement的状态都是unknown。
除此之外，参考资料有效的情况下，对于一个statement来说，如果它包含的事实或数据在参考资料中可以全部或部分找到，就认为它是supported的（数据接受四舍五入）；如果statement中所有的事实和数据在参考资料中都找不到，认为它是unsupported的。

你应该返回json列表格式，列表中的每一项包含statement的序号和判断结果，例如：
[
    {{
        "idx": 1,
        "result": "supported"
    }},
    {{
        "idx": 2,
        "result": "unsupported"
    }}
]

下面是参考资料和statements：
<reference>
{reference}
</reference>

<statements>
{statements}
</statements>

下面开始判断，直接输出json列表，不要输出任何闲聊或解释。"""


# ============================================================================
# Score Calculator
# ============================================================================
def calculate_weighted_scores(
    llm_output_json: dict, criteria_data: dict, language: str = "en"
) -> dict:
    """Calculate weighted scores based on LLM output and criteria weights."""
    results = {
        "target": {"dims": {}, "total": 0.0},
        "reference": {"dims": {}, "total": 0.0},
    }
    total_target_score = 0.0
    total_reference_score = 0.0

    dimension_weights = criteria_data.get("dimension_weight", {})
    task_id = criteria_data.get("id", "Unknown")

    if "criterions" not in criteria_data or not criteria_data["criterions"]:
        raise ValueError(
            f"ID: {task_id} - Missing required criterions data"
        )

    criterion_weights = {}
    for dim, criterions in criteria_data.get("criterions", {}).items():
        criterion_weights[dim] = {
            crit["criterion"]: crit["weight"] for crit in criterions
        }

    for dim, scores_list in llm_output_json.items():
        if not isinstance(scores_list, list):
            continue
        if dim not in dimension_weights or dim not in criterion_weights:
            continue

        dim_target_weighted_sum = 0.0
        dim_reference_weighted_sum = 0.0
        dim_total_weight = 0.0
        dim_criteria_map = criterion_weights.get(dim, {})
        if not dim_criteria_map:
            continue

        article_2_score = None
        for score_item in scores_list:
            if not isinstance(score_item, dict):
                continue

            criterion_text_raw = score_item.get("criterion")
            criterion_text = (
                criterion_text_raw.strip()
                if isinstance(criterion_text_raw, str)
                else None
            )

            article_1_score_raw = score_item.get("article_1_score")
            article_2_score_raw = score_item.get("article_2_score")
            target_score_raw = score_item.get("target_score")

            if target_score_raw is not None and article_1_score_raw is None:
                article_1_score_raw = target_score_raw

            try:
                article_1_score = (
                    float(article_1_score_raw)
                    if article_1_score_raw is not None
                    else None
                )
                article_2_score = (
                    float(article_2_score_raw)
                    if article_2_score_raw is not None
                    else None
                )
            except (ValueError, TypeError):
                continue

            if criterion_text and article_1_score is not None:
                weight = dim_criteria_map.get(criterion_text)

                # Fuzzy matching
                if weight is None:
                    criterion_lower = criterion_text.lower()
                    for key, val in dim_criteria_map.items():
                        if key.lower() == criterion_lower:
                            weight = val
                            break
                    if weight is None:
                        for key, val in dim_criteria_map.items():
                            if (
                                criterion_lower in key.lower()
                                or key.lower() in criterion_lower
                            ):
                                weight = val
                                break

                if weight is None:
                    weight = sum(dim_criteria_map.values()) / len(dim_criteria_map)

                dim_target_weighted_sum += article_1_score * weight
                dim_total_weight += weight
                if article_2_score is not None:
                    dim_reference_weighted_sum += article_2_score * weight

        if dim_total_weight > 0:
            dim_target_avg = dim_target_weighted_sum / dim_total_weight
            dim_reference_avg = (
                dim_reference_weighted_sum / dim_total_weight
                if article_2_score is not None
                else 0
            )
        else:
            dim_target_avg = 0
            dim_reference_avg = 0

        results["target"]["dims"][f"{dim}_weighted_avg"] = dim_target_avg
        results["reference"]["dims"][f"{dim}_weighted_avg"] = dim_reference_avg

        dim_weight = dimension_weights.get(dim, 0)
        total_target_score += dim_target_avg * dim_weight
        total_reference_score += dim_reference_avg * dim_weight

    results["target"]["total"] = total_target_score
    results["reference"]["total"] = total_reference_score
    return results


# ============================================================================
# Article Cleaner
# ============================================================================
class ArticleCleaner:
    """Clean articles by removing citations and references."""

    def __init__(self, clean_agent: GeminiClient):
        self.clean_agent = clean_agent
        self.min_valid_length = 100

    def _clean_text(self, text: str, language: str = "en", max_retries: int = 3) -> Optional[str]:
        prompt_template = CLEAN_PROMPT_ZH if language == "zh" else CLEAN_PROMPT_EN
        user_prompt = prompt_template.format(article=text)

        for retry in range(max_retries):
            try:
                result = self.clean_agent.generate(user_prompt=user_prompt, system_prompt="")
                if result and len(result.strip()) >= self.min_valid_length:
                    return result
            except Exception as e:
                logger.error(f"API call error: {e}")
                error_str = str(e).lower()
                if "tokens" in error_str and "less than" in error_str:
                    return None
        return None

    def chunk_clean_article(self, article: str, language: str = "en") -> Optional[str]:
        """Split long article into two chunks for processing."""
        logger.info("Attempting to process article in 2 chunks")
        chunk_size = len(article) // 2
        chunks = []

        # First chunk: split at sentence boundary
        end = chunk_size
        search_start = max(0, end - 200)
        for j in range(end, search_start, -1):
            if j < len(article) and article[j] in ".?!。？！\n":
                end = j + 1
                break
        chunks.append(article[:end])
        chunks.append(article[end:])

        cleaned_chunks = []
        for i, chunk in enumerate(chunks):
            clean_result = self._clean_text(chunk, language)
            if clean_result is None and len(chunk) > 200000:
                return None
            cleaned_chunks.append(clean_result or "")

        return "".join(cleaned_chunks)

    def clean_single(
        self,
        item: dict,
        output_file: str = None,
        processed_ids: set = None,
        file_lock=None,
        pbar_lock=None,
        pbar=None,
        max_retries: int = 5,
        language: str = "en",
    ):
        """Clean a single article."""
        try:
            data = item.copy()
            item_id = data.get("id")
            prompt = data.get("prompt", "")
            article = data.get("article", "")

            if not item_id or not prompt or not article:
                if pbar and pbar_lock:
                    with pbar_lock:
                        pbar.update(1)
                return None

            if processed_ids is not None and item_id in processed_ids:
                if pbar and pbar_lock:
                    with pbar_lock:
                        pbar.update(1)
                return None

            cleaned_article = self._clean_text(article, language, max_retries)
            if cleaned_article is None:
                cleaned_article = self.chunk_clean_article(article, language=language)

            if not cleaned_article or len(cleaned_article.strip()) < self.min_valid_length:
                if pbar and pbar_lock:
                    with pbar_lock:
                        pbar.update(1)
                return {"id": item_id, "error": "Failed to clean article"}

            result = {"id": item_id, "prompt": prompt, "article": cleaned_article}

            if output_file and file_lock and processed_ids is not None:
                with file_lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed_ids.add(item_id)
                if pbar and pbar_lock:
                    with pbar_lock:
                        pbar.update(1)
                return item_id
            else:
                if pbar and pbar_lock:
                    with pbar_lock:
                        pbar.update(1)
                return result

        except Exception as e:
            if pbar and pbar_lock:
                with pbar_lock:
                    pbar.update(1)
            logger.error(f"Error cleaning article {item.get('id', 'unknown')}: {e}")
            return None

    def clean_articles(
        self,
        model_name: str,
        raw_data_dir: str,
        cleaned_data_dir: str,
        max_workers: int = 5,
        max_retries: int = 5,
        limit: int = None,
        language: str = "en",
    ):
        """Clean articles for a model."""
        from tqdm import tqdm

        os.makedirs(cleaned_data_dir, exist_ok=True)
        input_file = os.path.join(raw_data_dir, f"{model_name}.jsonl")
        output_file = os.path.join(cleaned_data_dir, f"{model_name}.jsonl")

        if not os.path.exists(input_file):
            logger.warning(f"Input file not found: {input_file}")
            return

        logger.info(f"=== Cleaning {model_name} articles ===")

        # Load input data
        all_items = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_items.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if limit is not None and limit > 0:
            all_items = all_items[:limit]

        # Load processed IDs
        processed_ids = set()
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "id" in data:
                                processed_ids.add(data["id"])
                        except json.JSONDecodeError:
                            pass

        to_process = [
            item for item in all_items if item.get("id") not in processed_ids
        ]
        logger.info(
            f"Total: {len(all_items)}, to process: {len(to_process)}, already done: {len(processed_ids)}"
        )

        if not to_process:
            logger.info("All items already processed.")
            return

        file_lock = threading.Lock()
        pbar_lock = threading.Lock()

        with tqdm(
            total=len(all_items),
            desc=f"Cleaning {model_name}",
            initial=len(processed_ids),
        ) as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(
                        self.clean_single,
                        item,
                        output_file,
                        processed_ids,
                        file_lock,
                        pbar_lock,
                        pbar,
                        max_retries,
                        language,
                    )
                    for item in to_process
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()


# ============================================================================
# DRB Formatter (from drb_formatter.py)
# ============================================================================
def parse_xml_snippets(text: str) -> dict:
    """Parse XML-like snippets into a dictionary."""
    pattern = re.compile(
        r"<snippet id=([^>]+)>"
        r"\s*Title: (.*?)"
        r"\s*(?:URL: (.*?))?"
        r"\s*Snippet: (.*?)"
        r"\s*</snippet>",
        re.DOTALL,
    )
    matches = pattern.findall(text)
    results = {}
    for match in matches:
        snippet_id = match[0].strip()
        title = match[1].strip()
        url = match[2].strip() if match[2] else ""
        snippet = match[3].strip()
        results[snippet_id] = {"Title": title, "URL": url, "Snippet": snippet}
    return results


def parse_and_format_citations(text: str):
    """Parse <cite> tags and reformat the string."""
    citation_ids = []
    id_to_text_map = {}

    def replacer(match):
        ids_string = match.group(1)
        text_content = match.group(2)
        ids = ids_string.split(",")
        for this_id in ids:
            if this_id not in citation_ids:
                citation_ids.append(this_id)
        for cid in ids:
            if cid not in id_to_text_map:
                id_to_text_map[cid] = [text_content]
            else:
                id_to_text_map[cid].append(text_content)
        formatted_ids = " ".join([f"[{id}]" for id in ids])
        return f". {text_content} {formatted_ids}."

    pattern = re.compile(r'<cite id="([^"]+)">([^<]+)</cite>')
    formatted_text = pattern.sub(replacer, text)
    return formatted_text, citation_ids, id_to_text_map


def parse_search_results(text: str) -> list:
    """Parse search results from text."""
    pattern = re.compile(
        r"Title: (.*?)\n"
        r"(?:URL: (.*?)\n)?"
        r"Snippet: (.*?)"
        r"(?=\nTitle: |\Z)",
        re.DOTALL,
    )
    matches = pattern.findall(text)
    results = []
    for match in matches:
        results.append(
            {
                "Title": match[0].strip(),
                "URL": match[1].strip(),
                "Snippet": match[2].strip(),
            }
        )
    return results


def format_drb_data(example: dict) -> dict:
    """Convert a single example to DRB format."""
    output = {}
    output["id"] = example.get("example_id", example.get("id"))
    output["prompt"] = example["problem"]
    output["article"] = ""
    output["citations_deduped"] = {}
    output["citations"] = []

    traces = {}
    all_urls = {}
    if "full_traces" in example:
        for call in example["full_traces"]["tool_calls"]:
            if "call_id" in call:
                call_id = call["call_id"]
                call_results = parse_search_results(call["output"])
                if len(call_results) == 0:
                    try:
                        aggregated_results = "\n\n".join(
                            raw_call["output"].strip()
                            for raw_call in call["raw_output"]["tool_outputs"]
                        )
                        call_results = parse_search_results(aggregated_results)
                    except Exception:
                        pass

                for i, each_result in enumerate(call_results):
                    traces[call_id + "-" + str(i)] = each_result
                    all_urls[each_result["URL"]] = call_id + "-" + str(i)
            else:
                snippets = parse_xml_snippets(call["generated_text"])
                for snippet_id, snippet_content in snippets.items():
                    traces[snippet_id] = snippet_content
                    all_urls[snippet_content["URL"]] = snippet_id

    if "final_response" not in example:
        if "pred_answer" in example:
            example["final_response"] = example["pred_answer"]

    article, citation_ids, id_to_text_map = parse_and_format_citations(
        example["final_response"]
    )

    for url, cid in all_urls.items():
        if cid in id_to_text_map:
            output["citations_deduped"][url] = {
                "facts": id_to_text_map[cid],
                "url_content": traces[cid]["Title"] + "\n\n" + traces[cid]["Snippet"],
            }
            for fact in id_to_text_map[cid]:
                output["citations"].append(
                    {"fact": fact, "ref_indx": cid, "url": url}
                )

    reference = []
    for i, each_id in enumerate(citation_ids):
        if each_id in traces:
            reference.append("[" + str(i + 1) + "] " + traces[each_id]["URL"])
            article = article.replace(each_id, str(i + 1))
        else:
            indicator = False
            for each_url in all_urls:
                if each_id in each_url:
                    reference.append("[" + str(i + 1) + "] " + each_url)
                    article = article.replace(each_id, str(i + 1))
                    indicator = True
                    break
            if not indicator:
                article = article.replace("[" + each_id + "]", "")

    if contains_chinese(article):
        output["article"] = article + "\n\n 参考文献: " + "\n".join(reference)
    else:
        output["article"] = article + "\n\n References: " + "\n".join(reference)

    return output


def run_format_conversion(input_file: str, task_name: str, output_dir: str) -> tuple:
    """
    Run format conversion for DRB evaluation.

    Returns:
        (raw_data_path, scraped_data_path): Paths to the formatted output files
    """
    data = load_jsonl(input_file)

    raw_data_dir = os.path.join(output_dir, "raw_data")
    scrape_dir = os.path.join(output_dir, "fact", task_name)
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(scrape_dir, exist_ok=True)

    formatted_data = []
    for i, item in enumerate(data):
        try:
            formatted_data.append(format_drb_data(item))
        except Exception:
            logger.warning(f"Error formatting data point {i}")

    raw_data_path = os.path.join(raw_data_dir, f"{task_name}.jsonl")
    with open(raw_data_path, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")

    scraped_data_path = os.path.join(scrape_dir, "scraped.jsonl")
    with open(scraped_data_path, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")

    logger.info(f"Formatted {len(formatted_data)} items -> {raw_data_path}")
    return raw_data_path, scraped_data_path


# ============================================================================
# RACE Evaluation
# ============================================================================
def format_criteria_list(criteria_data: dict) -> str:
    """Format evaluation criteria list as JSON string."""
    criteria_for_prompt = {}
    criterions_dict = criteria_data.get("criterions", {})
    for dim, criterions_list in criterions_dict.items():
        if not isinstance(criterions_list, list):
            continue
        criteria_for_prompt[dim] = []
        for crit_item in criterions_list:
            if isinstance(crit_item, dict) and "criterion" in crit_item and "explanation" in crit_item:
                criteria_for_prompt[dim].append(
                    {
                        "criterion": crit_item["criterion"],
                        "explanation": crit_item["explanation"],
                    }
                )
    return json.dumps(criteria_for_prompt, ensure_ascii=False, indent=2)


def process_single_race_item(
    task_data,
    target_articles_map,
    reference_articles_map,
    criteria_map,
    llm_client,
    lock,
    pbar,
    max_retries,
    language,
):
    """Process a single RACE evaluation item."""
    task_id = task_data.get("id")
    prompt = task_data.get("prompt")

    if prompt not in target_articles_map:
        with lock:
            pbar.update(1)
        return {"id": task_id, "prompt": prompt, "error": "Target article not found"}

    if prompt not in reference_articles_map:
        with lock:
            pbar.update(1)
        return {"id": task_id, "prompt": prompt, "error": "Reference article not found"}

    if prompt not in criteria_map:
        with lock:
            pbar.update(1)
        return {"id": task_id, "prompt": prompt, "error": "Criteria not found"}

    target_article = target_articles_map[prompt].get("article", "")
    reference_article = reference_articles_map[prompt].get("article", "")

    try:
        criteria_list_str = format_criteria_list(criteria_map[prompt])
    except ValueError as e:
        with lock:
            pbar.update(1)
        return {"id": task_id, "prompt": prompt, "error": str(e)}

    score_prompt = SCORE_PROMPT_ZH if language == "zh" else SCORE_PROMPT_EN
    user_prompt = score_prompt.format(
        task_prompt=prompt,
        article_1=target_article,
        article_2=reference_article,
        criteria_list=criteria_list_str,
    )

    llm_response_str = None
    llm_output_json = None
    success = False
    retry_count = 0

    while retry_count < max_retries and not success:
        try:
            llm_response_str = llm_client.generate(
                user_prompt=user_prompt, system_prompt=""
            )
            json_str_extracted = extract_json_from_markdown(llm_response_str)
            if not json_str_extracted:
                raise ValueError("Failed to extract JSON from LLM response")
            llm_output_json = json.loads(json_str_extracted)
            expected_dims = [
                "comprehensiveness",
                "insight",
                "instruction_following",
                "readability",
            ]
            if not all(dim in llm_output_json for dim in expected_dims):
                missing = [dim for dim in expected_dims if dim not in llm_output_json]
                raise ValueError(f"Missing dimensions: {missing}")
            success = True
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(1.5 ** retry_count)
            else:
                logger.error(
                    f"ID {task_id}: Failed after {max_retries} retries - {str(e)}"
                )

    if not success:
        with lock:
            pbar.update(1)
        return {
            "id": task_id,
            "prompt": prompt,
            "error": f"Failed after {max_retries} retries",
        }

    try:
        scores = calculate_weighted_scores(
            llm_output_json, criteria_map[prompt], language
        )
        target_total = scores["target"]["total"]
        reference_total = scores["reference"]["total"]
        overall_score = (
            target_total / (target_total + reference_total)
            if target_total + reference_total > 0
            else 0
        )

        normalized_dims = {}
        for dim in [
            "comprehensiveness",
            "insight",
            "instruction_following",
            "readability",
        ]:
            dim_key = f"{dim}_weighted_avg"
            if dim_key in scores["target"]["dims"]:
                t = scores["target"]["dims"][dim_key]
                r = scores["reference"]["dims"][dim_key]
                normalized_dims[dim] = t / (t + r) if t + r > 0 else 0
            else:
                normalized_dims[dim] = 0

    except Exception as e:
        with lock:
            pbar.update(1)
        return {"id": task_id, "prompt": prompt, "error": f"Scoring error: {str(e)}"}

    final_result = {
        "id": task_id,
        "prompt": prompt,
        "comprehensiveness": normalized_dims.get("comprehensiveness", 0),
        "insight": normalized_dims.get("insight", 0),
        "instruction_following": normalized_dims.get("instruction_following", 0),
        "readability": normalized_dims.get("readability", 0),
        "overall_score": overall_score,
    }

    with lock:
        pbar.update(1)
    return final_result


def run_race_evaluation(
    task_name: str,
    output_dir: str,
    llm_client: GeminiClient,
    max_workers: int = 5,
    limit: int = None,
    only_zh: bool = False,
    only_en: bool = False,
    force: bool = False,
    skip_cleaning: bool = False,
):
    """Run RACE evaluation."""
    from tqdm import tqdm

    raw_data_dir = os.path.join(output_dir, "raw_data")
    cleaned_data_dir = os.path.join(output_dir, "cleaned_data")
    race_output_dir = os.path.join(output_dir, "race", task_name)
    os.makedirs(race_output_dir, exist_ok=True)
    os.makedirs(cleaned_data_dir, exist_ok=True)

    output_file = os.path.join(race_output_dir, "raw_results.jsonl")
    result_file = os.path.join(race_output_dir, "race_result.txt")

    # Check existing results
    existing_results = []
    existing_ids = set()
    if os.path.exists(output_file) and not force:
        existing_results = load_jsonl(output_file)
        existing_ids = {r.get("id") for r in existing_results if r.get("id")}
        logger.info(f"Found {len(existing_results)} existing results")

    all_results = list(existing_results)

    # Process each language
    languages = []
    if not only_en:
        languages.append("zh")
    if not only_zh:
        languages.append("en")

    for language in languages:
        logger.info(f"Processing {language} data...")

        # Article cleaning
        if not skip_cleaning:
            cleaner = ArticleCleaner(llm_client)
            cleaner.clean_articles(
                task_name, raw_data_dir, cleaned_data_dir, max_workers, 5, limit, language
            )

        # Load all data
        all_tasks = load_jsonl(str(QUERY_FILE))
        all_tasks = [t for t in all_tasks if t.get("language") == language]
        if limit is not None and limit > 0:
            all_tasks = all_tasks[:limit]

        task_prompts = {t["prompt"] for t in all_tasks if "prompt" in t}
        all_criteria = load_jsonl(str(CRITERIA_FILE))
        criteria_list = [c for c in all_criteria if c.get("prompt") in task_prompts]

        target_file = os.path.join(cleaned_data_dir, f"{task_name}.jsonl")
        if not os.path.exists(target_file):
            logger.warning(f"Cleaned target file not found: {target_file}")
            continue

        all_target = load_jsonl(target_file)
        target_list = [a for a in all_target if a.get("prompt") in task_prompts]
        if not target_list:
            logger.warning(f"No target articles found for {language}")
            continue

        all_reference = load_jsonl(str(REFERENCE_FILE))
        reference_list = [a for a in all_reference if a.get("prompt") in task_prompts]

        # Build maps
        criteria_map = {item["prompt"]: item for item in criteria_list}
        target_map = {item["prompt"]: item for item in target_list}
        reference_map = {item["prompt"]: item for item in reference_list}

        tasks_to_process = [
            t
            for t in all_tasks
            if t.get("prompt") in criteria_map
            and t.get("prompt") in target_map
            and t.get("prompt") in reference_map
            and t.get("id") not in existing_ids
        ]

        if not tasks_to_process:
            logger.info(f"No tasks to process for {language}")
            continue

        logger.info(f"Processing {len(tasks_to_process)} {language} tasks...")

        lock = threading.Lock()
        with tqdm(
            total=len(tasks_to_process), desc=f"RACE {language} {task_name}"
        ) as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(
                        process_single_race_item,
                        task,
                        target_map,
                        reference_map,
                        criteria_map,
                        llm_client,
                        lock,
                        pbar,
                        MAX_RETRIES,
                        language,
                    )
                    for task in tasks_to_process
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        all_results.append(result)

    # Save results
    if all_results:
        all_results.sort(key=lambda x: x.get("id", float("inf")))
        with open(output_file, "w", encoding="utf-8") as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        successful = [r for r in all_results if "error" not in r]
        if successful:
            metrics = {}
            for key in [
                "comprehensiveness",
                "insight",
                "instruction_following",
                "readability",
                "overall_score",
            ]:
                metrics[key] = sum(r.get(key, 0) for r in successful) / len(successful)

            logger.info("\n=== RACE Evaluation Results ===")
            for key, val in metrics.items():
                logger.info(f"  {key}: {val:.4f}")
            logger.info("================================")

            with open(result_file, "w", encoding="utf-8") as f:
                for key, val in metrics.items():
                    f.write(f"{key}: {val:.4f}\n")

            return metrics

    return None


# ============================================================================
# FACT Evaluation
# ============================================================================
def validate_single_citation(data: tuple, id_to_lang_map: dict, llm_client: GeminiClient) -> dict:
    """Validate a single citation against its reference."""
    url = data[0]
    ref = data[1]["url_content"]
    facts = data[1]["facts"]
    article_id = data[1].get("article_id")

    if ref is None:
        return {"url": url, "validate_res": [], "error": "no reference"}

    if not article_id or article_id not in id_to_lang_map:
        return {"url": url, "validate_res": [], "error": "Language not found"}

    lang = id_to_lang_map[article_id]
    facts_str = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(facts)])

    if lang == "zh":
        user_prompt = FACT_VALIDATE_PROMPT_ZH.format(
            reference=ref, statements=facts_str
        )
    else:
        user_prompt = FACT_VALIDATE_PROMPT_EN.format(
            reference=ref, statements=facts_str
        )

    retries = 0
    error = None
    while retries < 3:
        try:
            response = llm_client.generate(
                user_prompt=user_prompt,
                model=FACT_MODEL,
            )
            validate_res = json.loads(
                response.replace("```json", "").replace("```", "")
            )
            for _v in validate_res:
                _v["idx"] -= 1
            assert len(validate_res) == len(facts)
            return {"url": url, "validate_res": validate_res, "error": None}
        except Exception as e:
            error = str(e)
            time.sleep(3)
            retries += 1

    return {"url": url, "validate_res": [], "error": error}


def run_fact_evaluation(
    task_name: str,
    output_dir: str,
    llm_client: GeminiClient,
    max_workers: int = 10,
):
    """Run FACT evaluation."""
    from tqdm import tqdm

    fact_dir = os.path.join(output_dir, "fact", task_name)
    scraped_path = os.path.join(fact_dir, "scraped.jsonl")
    validated_path = os.path.join(fact_dir, "validated.jsonl")
    result_path = os.path.join(fact_dir, "fact_result.txt")

    if not os.path.exists(scraped_path):
        logger.error(f"Scraped data not found: {scraped_path}")
        return None

    raw_data = load_jsonl(scraped_path)

    # Load query data for language info
    query_data = load_jsonl(str(QUERY_FILE))
    id_to_lang_map = {
        item["id"]: item.get("language")
        for item in query_data
        if "id" in item and "language" in item
    }

    if not id_to_lang_map:
        raise ValueError("No valid language information found in query data")

    # Check existing results
    if os.path.exists(validated_path):
        processed = [d["id"] for d in load_jsonl(validated_path)]
        data_to_process = [d for d in raw_data if d["id"] not in processed]
    else:
        data_to_process = raw_data

    logger.info(f"FACT: Processing {len(data_to_process)} instances...")

    for d in tqdm(data_to_process, desc="FACT Validation"):
        citations = [(k, v) for k, v in d["citations_deduped"].items()]
        article_id = d.get("id")
        if not article_id:
            continue

        for citation in citations:
            citation[1]["article_id"] = article_id

        results = []
        for citation in citations:
            results.append(
                validate_single_citation(citation, id_to_lang_map, llm_client)
            )

        for res in results:
            d["citations_deduped"][res["url"]]["validate_res"] = res["validate_res"]
            d["citations_deduped"][res["url"]]["validate_error"] = res["error"]

        with open(validated_path, "a+", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Compute statistics
    total_citations = 0
    total_valid_citations = 0
    total_num = 0

    all_validated = load_jsonl(validated_path)
    for d in all_validated:
        if not d.get("citations"):
            continue
        for c in d["citations_deduped"].values():
            if c.get("validate_error") is not None:
                continue
            for _c in c.get("validate_res", []):
                if _c["result"] != "unknown":
                    total_citations += 1
                    if _c["result"] == "supported":
                        total_valid_citations += 1
        total_num += 1

    if total_num > 0 and total_citations > 0:
        metrics = {
            "avg_citations_per_article": total_citations / total_num,
            "avg_valid_citations_per_article": total_valid_citations / total_num,
            "valid_rate": total_valid_citations / total_citations,
        }

        with open(result_path, "w") as f:
            for key, val in metrics.items():
                f.write(f"{key}: {val:.4f}\n")

        logger.info("\n=== FACT Evaluation Results ===")
        for key, val in metrics.items():
            logger.info(f"  {key}: {val:.4f}")
        logger.info("================================")

        return metrics
    else:
        logger.warning("No valid citations found for FACT evaluation")
        return None


# ============================================================================
# Main entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Self-contained Deep Research Bench (DRB) Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (format + RACE + FACT):
  python run_eval.py --input_file eval_output/drb.jsonl --task_name my_model

  # RACE only:
  python run_eval.py --input_file eval_output/drb.jsonl --task_name my_model --skip_fact

  # FACT only:
  python run_eval.py --input_file eval_output/drb.jsonl --task_name my_model --skip_race

  # With limit for testing:
  python run_eval.py --input_file eval_output/drb.jsonl --task_name my_model --limit 2
""",
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the DR Tulu output JSONL file",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Identifier for this evaluation run",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for evaluation output (default: same dir as input file)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Maximum number of concurrent workers (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)",
    )
    parser.add_argument(
        "--only_zh",
        action="store_true",
        help="Only process Chinese data",
    )
    parser.add_argument(
        "--only_en",
        action="store_true",
        help="Only process English data",
    )
    parser.add_argument(
        "--skip_race",
        action="store_true",
        help="Skip RACE evaluation",
    )
    parser.add_argument(
        "--skip_fact",
        action="store_true",
        help="Skip FACT evaluation",
    )
    parser.add_argument(
        "--skip_cleaning",
        action="store_true",
        help="Skip article cleaning step in RACE evaluation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if results exist",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.input_file), f"drb_eval_{args.task_name}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize data files (download from HF if needed)
    global CRITERIA_FILE, REFERENCE_FILE, QUERY_FILE
    CRITERIA_FILE, REFERENCE_FILE, QUERY_FILE = _ensure_data_files()

    logger.info(f"=== DRB Evaluation ===")
    logger.info(f"Input: {args.input_file}")
    logger.info(f"Task: {args.task_name}")
    logger.info(f"Output: {args.output_dir}")

    # Check API key
    if not os.environ.get("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable not set")
        return

    # Initialize Gemini client
    llm_client = GeminiClient(model=RACE_MODEL)

    # Step 1: Format conversion
    logger.info("\n--- Step 1: Format Conversion ---")
    raw_data_path, scraped_data_path = run_format_conversion(
        args.input_file, args.task_name, args.output_dir
    )
    logger.info(f"Raw data: {raw_data_path}")
    logger.info(f"Scraped data: {scraped_data_path}")

    # Step 2: RACE evaluation
    if not args.skip_race:
        logger.info("\n--- Step 2: RACE Evaluation ---")
        race_metrics = run_race_evaluation(
            task_name=args.task_name,
            output_dir=args.output_dir,
            llm_client=llm_client,
            max_workers=args.max_workers,
            limit=args.limit,
            only_zh=args.only_zh,
            only_en=args.only_en,
            force=args.force,
            skip_cleaning=args.skip_cleaning,
        )
    else:
        logger.info("\n--- Step 2: RACE Evaluation (SKIPPED) ---")

    # Step 3: FACT evaluation
    if not args.skip_fact:
        logger.info("\n--- Step 3: FACT Evaluation ---")
        fact_metrics = run_fact_evaluation(
            task_name=args.task_name,
            output_dir=args.output_dir,
            llm_client=llm_client,
            max_workers=args.max_workers,
        )
    else:
        logger.info("\n--- Step 3: FACT Evaluation (SKIPPED) ---")

    logger.info("\n=== DRB Evaluation Complete ===")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

