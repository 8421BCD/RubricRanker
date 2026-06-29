#!/usr/bin/env python3
"""
Self-contained SQA-CS-V2 evaluation script.

Runs the SQA evaluation pipeline without requiring the external agent-baselines
repository. This script wraps the `astabench` and `inspect_ai` packages.

Usage:
    # Step 1: Convert DR Tulu output to ASTA format
    python run_eval.py convert --input_file <path/to/sqa_output.jsonl>

    # Step 2: Run SQA evaluation on the converted file
    python run_eval.py eval --input_file <path/to/sqa_output_asta_format.jsonl>

    # Or run both steps in one command
    python run_eval.py run --input_file <path/to/sqa_output.jsonl>

    # With options
    python run_eval.py run --input_file <path/to/sqa_output.jsonl> \
        --scorer_model "google/gemini-2.5-flash" \
        --max_connections 16 \
        --split test

Environment Variables Required:
    GOOGLE_API_KEY: Google API key (for Gemini-based scoring)
    HF_TOKEN: Hugging Face token (can be dummy if data is local)

Dependencies:
    pip install astabench==0.3.1 inspect_ai datasets
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import DR Tulu format conversion from Varsha's script (single source of truth)
sys.path.insert(0, str(Path(__file__).parent))
from convert_to_asta_format import parse_answer


def convert_to_asta_format(input_file: str, output_file: str = None) -> str:
    """
    Convert DR Tulu output to ASTA format for SQA evaluation.

    Uses the DR Tulu citation format: <cite id="xxx">text</cite> in final_response
    with <snippet id="xxx">content</snippet> in tool call traces.

    Args:
        input_file: Path to the DR Tulu output JSONL file
        output_file: Path for the output file (default: input_file with _asta_format suffix)

    Returns:
        Path to the output file
    """
    if output_file is None:
        output_file = input_file.replace(".jsonl", "_asta_format.jsonl")

    print(f"Converting (DR Tulu format): {input_file}")

    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    formatted = []
    for i, sample in enumerate(data):
        try:
            parsed = parse_answer(sample)
            formatted.append(
                {
                    "question": sample["problem"],
                    "response": {"sections": parsed},
                }
            )
        except Exception as e:
            print(f"Warning: Failed to convert sample {i}: {e}")

    with open(output_file, "w") as f:
        json.dump(formatted, f, indent=4)

    print(f"Converted {len(formatted)}/{len(data)} samples -> {output_file}")
    return output_file


# ============================================================================
# SQA Evaluation (wrapping astabench/inspect)
# ============================================================================
def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    try:
        import astabench
    except ImportError:
        missing.append("astabench==0.3.1")

    try:
        import inspect_ai
    except ImportError:
        missing.append("inspect_ai")

    try:
        import datasets
    except ImportError:
        missing.append("datasets")

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True


def create_cached_solver_script(data_path: str, split: str = "test") -> str:
    """
    Create a temporary solver script that loads cached results.

    This replicates the behavior of agent_baselines/solvers/sqa/debug/cached_solver.py
    without needing the agent-baselines repo.
    """
    solver_code = f'''
import json
from pathlib import Path

from astabench.types.sqa import SQAResponse, SQAResponseWithUsage
from astabench.util.model import record_model_usage_with_inspect
from inspect_ai.model import ChatMessageAssistant, ModelUsage
from inspect_ai.solver import Generate, Solver, TaskState, solver
from datasets.utils.logging import set_verbosity_error

set_verbosity_error()
import os
from datasets import load_dataset

_dataset = None

def _load_dataset(path: str, split: str):
    global _dataset
    if _dataset is None:
        if os.path.exists(path):
            print(f"Loading data from local file: {{path}}")
            _dataset = load_dataset("json", data_files={{split: path}})[split]
        else:
            raise FileNotFoundError(f"Data file not found: {{path}}")

def _read_q_result(question: str):
    results = _dataset.filter(lambda x: x["question"] == question)
    if len(results) == 0:
        return None
    return results[0]["response"]

def _query_cache(question: str):
    response_dict = _read_q_result(question)
    if response_dict is None:
        raise ValueError(f"No results found for question: {{question}}")
    response = (
        SQAResponseWithUsage(**response_dict)
        if "tokens" in response_dict
        else SQAResponse(**response_dict)
    )
    return response

@solver
def cache_solver(
    path: str = "{data_path}",
    split: str = "{split}",
    model: str = "openai/gpt-4.1",
) -> Solver:
    _load_dataset(path, split)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        question = state.metadata["initial_prompt"]
        response = _query_cache(question)
        content = json.dumps(
            {{"response": response.model_dump(mode="json", exclude={{"tokens"}})}},
            indent=2,
        )
        if hasattr(response, "tokens"):
            usage_dict = {{f"{{k}}_tokens": v for k, v in response.tokens.items()}}
            record_model_usage_with_inspect(
                model_name=model,
                usage=ModelUsage(**usage_dict),
            )
        state.messages.append(ChatMessageAssistant(content=content))
        state.output.completion = content
        return state

    return solve
'''
    return solver_code


def _build_inspect_cmd(
    solver_path: str,
    input_file: str,
    split: str,
    scorer_model: str,
    max_connections: int,
    simplified_eval: bool,
    assess_jointly: bool,
    sentence_wise_cit_eval: bool,
    all_at_once: bool,
    with_search_tools: bool,
    output_dir: str = None,
) -> List[str]:
    """Build the inspect eval command arguments (without the python/uv prefix)."""
    args = [
        "eval", "astabench/sqa",
        "--display", "plain",
        "--solver", solver_path,
        "-S", f"path={os.path.abspath(input_file)}",
        "-T", f"split={split}",
        "-T", f"with_search_tools={with_search_tools}",
        "-T", f"simplified_eval={str(simplified_eval).lower()}",
        "-T", f"assess_jointly={str(assess_jointly).lower()}",
        "--max-connections", str(max_connections),
        "-T", f"sentence_wise_cit_eval={str(sentence_wise_cit_eval).lower()}",
        "-T", f"all_at_once={str(all_at_once).lower()}",
        "-T", f"scorer_model={scorer_model}",
    ]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        args.extend(["--log-dir", output_dir])
    return args


def _check_uv_available() -> bool:
    """Check if uv is available."""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def run_sqa_evaluation(
    input_file: str,
    split: str = "test",
    scorer_model: str = "google/gemini-2.5-flash",
    max_connections: int = 16,
    simplified_eval: bool = True,
    assess_jointly: bool = True,
    sentence_wise_cit_eval: bool = False,
    all_at_once: bool = True,
    with_search_tools: bool = False,
    output_dir: str = None,
):
    """
    Run SQA evaluation using astabench + inspect_ai.

    Automatically handles the Python 3.11+ requirement:
    - If current Python >= 3.11 and packages are installed: runs directly
    - Otherwise: uses `uv run` with a temporary project to manage dependencies

    Args:
        input_file: Path to the ASTA-formatted JSONL file
        split: Dataset split to evaluate on (test/dev)
        scorer_model: Model to use for scoring
        max_connections: Maximum concurrent connections
        simplified_eval: Use simplified evaluation
        assess_jointly: Assess jointly
        sentence_wise_cit_eval: Sentence-wise citation evaluation
        all_at_once: Process all at once
        with_search_tools: Whether to use search tools
        output_dir: Directory for evaluation output
    """
    # Check environment variables
    if not os.environ.get("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY not set. Required for Gemini-based scoring.")
        print("Set it with: export GOOGLE_API_KEY='your_key_here'")

    # Create temporary solver script
    solver_code = create_cached_solver_script(
        data_path=os.path.abspath(input_file), split=split
    )

    # Write to temporary directory
    solver_dir = tempfile.mkdtemp(prefix="sqa_eval_")
    solver_path = os.path.join(solver_dir, "cached_solver.py")
    with open(solver_path, "w") as f:
        f.write(solver_code)

    # Build inspect eval arguments
    inspect_args = _build_inspect_cmd(
        solver_path=solver_path,
        input_file=input_file,
        split=split,
        scorer_model=scorer_model,
        max_connections=max_connections,
        simplified_eval=simplified_eval,
        assess_jointly=assess_jointly,
        sentence_wise_cit_eval=sentence_wise_cit_eval,
        all_at_once=all_at_once,
        with_search_tools=with_search_tools,
        output_dir=output_dir,
    )

    # Determine execution method
    use_uv = False
    py_version = sys.version_info

    if py_version >= (3, 11) and check_dependencies():
        # Direct execution: Python >= 3.11 and deps installed
        cmd = [sys.executable, "-m", "inspect_ai._cli.main"] + inspect_args
    elif _check_uv_available():
        # Use uv run: manages Python version + deps automatically
        use_uv = True
        # Create a temporary pyproject.toml for uv
        pyproject_content = """
[project]
name = "sqa-eval-runner"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
    "astabench==0.3.1",
    "inspect_ai",
    "datasets",
]

[tool.uv]
override-dependencies = [
    "openai==1.78.0",
]
"""
        pyproject_path = os.path.join(solver_dir, "pyproject.toml")
        with open(pyproject_path, "w") as f:
            f.write(pyproject_content)

        cmd = [
            "uv", "run", "--python", "3.11",
            "inspect",
        ] + inspect_args
    else:
        print("ERROR: Cannot run SQA evaluation:")
        print(f"  - Current Python: {py_version.major}.{py_version.minor} (need >= 3.11)")
        print(f"  - astabench requires Python >= 3.11")
        print()
        print("Options:")
        print("  1. Install uv: pip install uv")
        print("  2. Use Python >= 3.11 with: pip install astabench==0.3.1 inspect_ai datasets")
        print("  3. Create a conda env: conda create -n sqa_eval python=3.11 && conda activate sqa_eval && pip install astabench==0.3.1 inspect_ai datasets")
        # Cleanup
        try:
            import shutil
            shutil.rmtree(solver_dir)
        except Exception:
            pass
        sys.exit(1)

    print(f"\n=== Running SQA Evaluation ===")
    print(f"Input file: {input_file}")
    print(f"Split: {split}")
    print(f"Scorer model: {scorer_model}")
    print(f"Max connections: {max_connections}")
    print(f"Method: {'uv run (auto-managed Python 3.11+)' if use_uv else 'direct'}")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Pass through environment variables, set dummy keys if needed
    env = os.environ.copy()
    # astabench entrypoint loading requires OPENAI_API_KEY even when not using OpenAI
    if not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = "dummy-not-used"
    # HF_TOKEN is needed for downloading gated allenai/asta-bench dataset
    if not env.get("HF_TOKEN"):
        # Try to get token from huggingface-cli login cache
        try:
            from huggingface_hub import HfFolder
            cached_token = HfFolder.get_token()
            if cached_token:
                env["HF_TOKEN"] = cached_token
                print(f"Using cached HF token from huggingface-cli login")
        except Exception:
            pass
    if not env.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN not set. Needed for gated allenai/asta-bench dataset.")
        print("  Run: huggingface-cli login")
        print("  Or set: export HF_TOKEN='your_token'")
        print("  And request access at: https://huggingface.co/datasets/allenai/asta-bench")

    # Run the evaluation
    result = subprocess.run(cmd, cwd=solver_dir, env=env)

    # Cleanup
    try:
        import shutil
        shutil.rmtree(solver_dir)
    except Exception:
        pass

    if result.returncode != 0:
        print(f"\nEvaluation failed with return code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"\n=== SQA Evaluation Complete ===")
        if output_dir:
            print(f"Logs saved to: {output_dir}")


# ============================================================================
# Main entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Self-contained SQA-CS-V2 Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert DR Tulu output to ASTA format:
  python run_eval.py convert --input_file eval_output/sqa.jsonl

  # Run evaluation on converted file:
  python run_eval.py eval --input_file eval_output/sqa_asta_format.jsonl

  # Full pipeline (convert + evaluate):
  python run_eval.py run --input_file eval_output/sqa.jsonl

  # With custom scorer model:
  python run_eval.py run --input_file eval_output/sqa.jsonl \\
      --scorer_model "google/gemini-2.5-flash"
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert DR Tulu output to ASTA format"
    )
    convert_parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the DR Tulu output JSONL file",
    )
    convert_parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path for the output file (default: auto-generated)",
    )

    # Eval command
    eval_parser = subparsers.add_parser(
        "eval", help="Run SQA evaluation on ASTA-formatted file"
    )
    eval_parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the ASTA-formatted JSONL file",
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    eval_parser.add_argument(
        "--scorer_model",
        type=str,
        default="google/gemini-2.5-flash",
        help="Scorer model (default: google/gemini-2.5-flash)",
    )
    eval_parser.add_argument(
        "--max_connections",
        type=int,
        default=16,
        help="Maximum concurrent connections (default: 16)",
    )
    eval_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for evaluation output logs",
    )

    # Run command (convert + eval)
    run_parser = subparsers.add_parser(
        "run", help="Full pipeline: convert + evaluate"
    )
    run_parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the DR Tulu output JSONL file",
    )
    run_parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    run_parser.add_argument(
        "--scorer_model",
        type=str,
        default="google/gemini-2.5-flash",
        help="Scorer model (default: google/gemini-2.5-flash)",
    )
    run_parser.add_argument(
        "--max_connections",
        type=int,
        default=16,
        help="Maximum concurrent connections (default: 16)",
    )
    run_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for evaluation output logs",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "convert":
        convert_to_asta_format(args.input_file, args.output_file)

    elif args.command == "eval":
        run_sqa_evaluation(
            input_file=args.input_file,
            split=args.split,
            scorer_model=args.scorer_model,
            max_connections=args.max_connections,
            output_dir=args.output_dir,
        )

    elif args.command == "run":
        # Step 1: Convert
        print("=== Step 1: Format Conversion ===")
        asta_file = convert_to_asta_format(args.input_file)

        # Step 2: Evaluate
        print("\n=== Step 2: SQA Evaluation ===")
        run_sqa_evaluation(
            input_file=asta_file,
            split=args.split,
            scorer_model=args.scorer_model,
            max_connections=args.max_connections,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()

