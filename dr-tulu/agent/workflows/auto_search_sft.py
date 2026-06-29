import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import dotenv
from dr_agent.agent_interface import BaseAgent
from dr_agent.client import DocumentToolOutput, LLMToolClient, ToolOutput
from dr_agent.shared_prompts import UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS
from dr_agent.tool_interface.chained_tool import ChainedTool
from dr_agent.tool_interface.mcp_tools import (
    BaseTool,
    Crawl4AIBrowseTool,
    HFPointwiseRerankerTool,
    JinaBrowseTool,
    SemanticScholarSnippetSearchTool,
    SerperBrowseTool,
    SerperSearchTool,
    VllmHostedListwiseRerankerTool,
    VllmHostedPointwiseRerankerTool,
    VllmHostedRerankerTool,
    VllmListwiseSinglePassRerankerTool,
    VllmListwiseSlidingWindowRerankerTool,
    VllmSetwiseRank4GenRerankerTool,
    VllmSetwiseSETRRerankerTool,
    VllmSetwiseRubricRankerTool,
)
from dr_agent.tool_interface.search_with_rerank_tool import SearchWithRerankTool
from dr_agent.utils import (
    check_port,
    extract_port_from_url,
    launch_mcp_server,
    launch_vllm_server,
)
from dr_agent.workflow import BaseWorkflow, BaseWorkflowConfiguration
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

# Make sure the .env file is in the root directory of the project rl-rag-mcp/.env
dotenv.load_dotenv(Path(__file__).parent.parent.parent / ".env")

@dataclass
class WebPageReaderAgentV2(BaseAgent):
    question: Optional[str] = None
    prompt = """
We are searching on the internet for the following question:
{question}
Here is some webpage scraped from the internet:
{document}
Can you clean the raw webpage text and convert it into a more readable format? You should remove all the unnecessary information and keep the main content of the page. Please produce the output in the format of "Cleaned webpage text:\n[you text here]".
""".strip()

    def preprocess_input(self, documents: Union[str, Any]) -> Dict[str, str]:
        # Accept either a raw string or a ToolOutput-like object with an `output` attribute
        assert self.question is not None, "Question is not set"

        if isinstance(documents, DocumentToolOutput):
            # print("using DocumentToolOutput")
            doc_str = "\n".join(
                [
                    document.simple_stringify()[: 32000 * 4 // len(documents.documents)]
                    for document in documents.documents
                ]
            )
        elif hasattr(documents, "output"):
            doc_str = documents.output
        else:
            doc_str = documents if isinstance(documents, str) else str(documents)
        input_params = {"question": self.question, "document": doc_str}
        # print(input_params)
        return input_params

    def postprocess_output(self, result: Dict[str, Any]) -> str:
        output_string = result.generated_text
        if "</think>" in output_string:
            output_string = "".join(output_string.split("</think>")[1:]).strip()

        if "Cleaned webpage text:" in output_string:
            output_string = output_string.split("Cleaned webpage text:")[1].strip()

        return output_string


@dataclass
class SearchAgent(BaseAgent):
    prompt_version: str = "v20250907"

    def prompt(
        self,
        question: str,
        dataset_name: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        PROMPT = UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS[self.prompt_version]
        system_prompt = PROMPT["system_prompt"]

        if dataset_name in [
            "2wiki",
            "simpleqa",
            "browsecomp",
            "bc_synthetic_depth_one_v2_verified",
            "bc_synthetic_varied_depth_o3_verified",
            "webwalker",
            "hle",
            "dsqa",
        ]:
            instruction_field_name = "exact_answer"
        elif dataset_name in ["sqav2", "genetic_diseases_qa"]:
            instruction_field_name = "long_form"
        elif dataset_name in ["healthbench", "deep_research_bench", "researchqa"]:
            instruction_field_name = "short_form"
        elif dataset_name and "sft-mix" in dataset_name:
            if "short_form" in dataset_name:
                instruction_field_name = "exact_answer"
            elif "long_form" in dataset_name:
                instruction_field_name = "long_form"  # or "short_form"?
            else:
                raise ValueError(
                    f"Unclear which instruction field name to use for the sft mix dataset: {dataset_name}"
                )
        else:
            if "exact_answer" in str(dataset_name):
                instruction_field_name = "exact_answer"
            elif "short_form" in str(dataset_name):
                instruction_field_name = "short_form"
            elif "long_form" in str(dataset_name):
                instruction_field_name = "long_form"
            else:
                print("set additional instructions none")
                instruction_field_name = None

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if history:
            messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": (
                    question
                    + "\n\n"
                    + PROMPT["additional_instructions"][instruction_field_name]
                    if instruction_field_name is not None
                    else question
                ),
            }
        )

        return messages

    def postprocess_output(self, result: Dict[str, Any]) -> str:
        output_string = result.generated_text
        if "</think>" in output_string:
            output_string = "".join(output_string.split("</think>")[1:]).strip()

        if "<answer>" in output_string:
            output_string = (
                output_string.split("<answer>")[1].split("</answer>")[0].strip()
            )

        # Replace the "\boxed{" with "\\boxed{"
        output_string = output_string.replace("\boxed{", "\\boxed{")

        if "\\boxed{" in output_string:
            output_string = output_string.split("\\boxed{")[1].split("}")[0].strip()

        return output_string


@dataclass
class AnswerAgent(BaseAgent):
    prompt_version: str = "v20250907"

    def prompt(self, question: str, history: str, dataset_name: str) -> str:

        PROMPT = UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS[self.prompt_version]
        if dataset_name in [
            "2wiki",
            "simpleqa",
            "browsecomp",
            "bc_synthetic_depth_one_v2_verified",
            "bc_synthetic_varied_depth_o3_verified",
            "webwalker",
            "webshaper",
        ]:
            instruction_field_name = "exact_answer"
        elif dataset_name in ["sqav2", "genetic_diseases_qa"]:
            instruction_field_name = "long_form"
        elif dataset_name in ["healthbench", "deep_research_bench", "researchqa"]:
            instruction_field_name = "short_form"
        else:
            if "exact_answer" in str(dataset_name):
                instruction_field_name = "exact_answer"
            elif "short_form" in str(dataset_name):
                instruction_field_name = "short_form"
            elif "long_form" in str(dataset_name):
                instruction_field_name = "long_form"
            else:
                print(f"AnswerAgent: set additional instructions to exact_answer as fallback for dataset: {dataset_name}")
                instruction_field_name = "exact_answer"

        return [
            {
                "role": "system",
                "content": PROMPT["system_prompt"],
            },
            {
                "role": "user",
                "content": question
                + "\n\n"
                + PROMPT["additional_instructions"][instruction_field_name],
            },
            {
                "role": "assistant",
                "content": history,
            },
            {
                "role": "user",
                "content": "Now please generate an answer based on the search results by far.",
            },
        ]

    def postprocess_output(self, result: Dict[str, Any]) -> str:
        output_string = result.generated_text
        if "</think>" in output_string:
            output_string = "".join(output_string.split("</think>")[1:]).strip()

        if "<answer>" in output_string:
            output_string = (
                output_string.split("<answer>")[1].split("</answer>")[0].strip()
            )

        # Replace the "\boxed{" with "\\boxed{"
        output_string = output_string.replace("\boxed{", "\\boxed{")

        if "\\boxed{" in output_string:
            output_string = output_string.split("\\boxed{")[1].split("}")[0].strip()

        return output_string


class NoBrowseTool(BaseTool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def __call__(self, *args, **kwargs):
        return DocumentToolOutput(
            output="Browse tool is not available at this time. Please try other tools.",
            called=True,
            timeout=False,
            runtime=0.0,
            error=None,
            call_id=self._generate_call_id(),
            raw_output=None,
            documents=[],
            tool_name="no_browse",
        )

    def _format_output(self, output: ToolOutput) -> str:
        return output.output

    def _generate_tool_schema(self):
        return {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "URL to browse"}},
            "required": ["url"],
        }


class _TopNTruncateSearchTool(BaseTool):
    """Thin wrapper that truncates an underlying search tool's output to top_n docs."""

    def __init__(self, search_tool: BaseTool, top_n: int):
        super().__init__(tool_parser=getattr(search_tool, "tool_parser", None),
                         name=getattr(search_tool, "name", None))
        self.search_tool = search_tool
        self.top_n = top_n

    async def __call__(self, tool_input):
        output = await self.search_tool(tool_input)
        if not isinstance(output, DocumentToolOutput) or self.top_n <= 0:
            return output
        if not output.documents or len(output.documents) <= self.top_n:
            return output
        truncated_docs = output.documents[: self.top_n]
        new_output_str = "\n\n".join(d.simple_stringify() for d in truncated_docs)
        return DocumentToolOutput(
            tool_name=output.tool_name,
            output=new_output_str if output.output else output.output,
            called=output.called,
            error=output.error,
            timeout=output.timeout,
            runtime=output.runtime,
            call_id=output.call_id,
            raw_output=output.raw_output,
            documents=truncated_docs,
            query=output.query,
        )

    def _format_output(self, output: ToolOutput) -> str:
        return self.search_tool._format_output(output)

    def _generate_tool_schema(self):
        return self.search_tool._generate_tool_schema()


class AutoReasonSearchWorkflow(BaseWorkflow):
    _default_configuration_path = os.path.join(
        os.path.dirname(__file__), "auto_search_sft.yaml"
    )

    class Configuration(BaseWorkflowConfiguration):

        tool_parser: str

        search_tool_name: str = "serper"

        # Separate generation client (SFT model)
        search_agent_base_url: Optional[str] = None
        search_agent_model_name: str = "dr-tulu/DR-Tulu-8B"
        search_agent_tokenizer_name: str = "Qwen/Qwen3-8B"
        search_agent_api_key: str = "dummy-key"
        search_agent_max_tokens: int = 32000
        search_agent_temperature: float = 0.7
        search_agent_max_tool_calls: int = 10

        use_browse_agent: bool = False
        browse_agent_base_url: Optional[str] = None
        browse_agent_model_name: str = "Qwen/Qwen3-8B"
        browse_agent_tokenizer_name: str = "Qwen/Qwen3-8B"
        browse_agent_api_key: str = "dummy-key"
        browse_agent_max_tokens: int = 32000
        browse_agent_temperature: float = 0.3

        # MCP transport configuration
        mcp_transport_type: str = "StreamableHttpTransport"
        mcp_executable: Optional[str] = None
        mcp_port: int = 8000

        # Search configuration
        number_documents_to_search: int = 10
        # How many docs to finally return to the agent (post-rerank or post-search).
        # Setwise rerankers (rank4gen/setr) ignore this and return their selected set.
        return_topn: int = 5
        search_timeout: int = 60

        # Browse configuration
        browse_tool_name: Optional[str] = "crawl4ai"
        browse_timeout: int = 60
        browse_max_pages_to_fetch: int = 10
        browse_context_char_length: int = 6000
        crawl4ai_use_docker_version: bool = False
        crawl4ai_use_ai2_config: bool = False

        prompt_version: str = "v20250907"

        # Reranker configuration
        use_reranker: bool = False
        # Supported types:
        #   pointwise_bge | pointwise_monot5 | pointwise_rankt5
        #   listwise_rankvicuna | listwise_rankzephyr
        #   setwise_rank4gen | setwise_setr | setwise_rubricranker
        #   listwise_singlepass
        #   listwise_openai | listwise_openai_rubrics
        #   pointwise / listwise (legacy aliases)
        reranker_type: str = "pointwise"
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
        reranker_api_url: Optional[str] = None  # vLLM reranker service URL
        # Deprecated; kept for backward compatibility. If set, overrides return_topn.
        reranker_top_n: int = -1
        # Deprecated; kept for backward compatibility. If set, overrides number_documents_to_search.
        reranker_num_search_results: int = -1
        reranker_doc_max_words: int = 300
        reranker_browse_max_pages: int = 50
        reranker_browse_timeout: int = 300
        reranker_timeout: int = 600
        reranker_listwise_max_tokens: int = 1024
        reranker_skip_browse: bool = False
        reranker_append_page_content_to_snippet: bool = False
        # HF backend (monoT5/RankT5) only.
        reranker_hf_device: str = "cuda:0"
        # Sliding-window listwise reranker (RankVicuna/RankZephyr) only.
        reranker_window_size: int = 20
        reranker_step: int = 10
        # Rank4Gen only.
        reranker_rank4gen_downstream_model: str = "default"
        reranker_rank4gen_downstream_desc: str = "default"
        # ---- OpenAI (Tencent data_eval) listwise reranker settings ----
        reranker_openai_model: str = "api_openai_gpt-5.1-response"
        reranker_openai_host: Optional[str] = None
        reranker_openai_port: Optional[int] = None
        reranker_openai_web_search: bool = False
        reranker_openai_max_retries: int = 3
        # ---- OpenAI rubrics-guided listwise reranker settings ----
        reranker_openai_rubrics_model: str = "api_openai_gpt-5.1-response"
        # ---- RubricRanker (student) setwise reranker settings ----
        # Prompt variant from rubricranker.yaml: "deepresearch" or "rag".
        reranker_rubricranker_prompt_variant: str = "deepresearch"
        # Sampling params (Qwen-style chat defaults).
        reranker_rubricranker_temperature: float = 0.7
        reranker_rubricranker_top_p: float = 0.8
        reranker_rubricranker_top_k: int = 20
        reranker_rubricranker_min_p: float = 0.0
        # ---- Single-pass listwise reranker (RankQwen-style) settings ----
        # Sampling params (Qwen-style chat defaults).
        reranker_singlepass_temperature: float = 0.7
        reranker_singlepass_top_p: float = 0.8
        reranker_singlepass_top_k: int = 20
        reranker_singlepass_min_p: float = 0.0

    def before_launch_check(self) -> None:
        """Check if MCP server and vLLM servers are running, launch if needed."""
        cfg = self.configuration
        if cfg is None:
            return

        console = Console()

        console.print()
        console.print(Panel.fit("🔍 Service Check", style="bold cyan"))
        console.print()

        # Check MCP server
        mcp_port = getattr(cfg, "mcp_port", 8000)
        if not check_port(mcp_port):
            console.print(
                f"[yellow]⚠[/yellow]  MCP server is not running on port [bold]{mcp_port}[/bold]"
            )
            if Confirm.ask("Launch MCP server?"):
                process = launch_mcp_server(mcp_port, self.logger)
                if process:
                    self._launched_processes.append(process)
                    console.print(
                        f"[green]✓[/green]  MCP server launched on port {mcp_port}"
                    )
                else:
                    console.print(
                        "[red]✗[/red]  Failed to start MCP server", style="bold red"
                    )
                    raise RuntimeError(
                        "Failed to start MCP server. Please launch it manually."
                    )
            else:
                console.print("[red]✗[/red]  MCP server is required", style="bold red")
                raise RuntimeError(
                    "MCP server is required. Please launch it manually or allow automatic launch."
                )
        else:
            console.print(
                f"[green]✓[/green]  MCP server is running on port [bold]{mcp_port}[/bold]"
            )

        # Check search agent vLLM server
        search_base_url = getattr(cfg, "search_agent_base_url", None)
        if search_base_url:
            port = extract_port_from_url(search_base_url)
            if port and not check_port(port):
                console.print(
                    f"[yellow]⚠[/yellow]  Search agent vLLM server is not running on port [bold]{port}[/bold]"
                )
                search_model = getattr(cfg, "search_agent_model_name", None)
                if search_model:
                    if Confirm.ask(
                        f"Launch vLLM server for [cyan]{search_model}[/cyan] on port {port}?"
                    ):
                        process = launch_vllm_server(
                            search_model, port, gpu_id=0, logger=self.logger
                        )
                        if process:
                            self._launched_processes.append(process)
                            console.print(
                                f"[green]✓[/green]  vLLM server launched for {search_model} on port {port}"
                            )
                        else:
                            console.print(
                                f"[yellow]⚠[/yellow]  Failed to start vLLM server. Manual launch command:"
                            )
                            console.print(
                                f"   [dim]CUDA_VISIBLE_DEVICES=0 vllm serve {search_model} --port {port} --dtype auto --max-model-len 40960[/dim]"
                            )
                    else:
                        console.print(f"[blue]💡[/blue]  Manual launch command:")
                        console.print(
                            f"   [dim]CUDA_VISIBLE_DEVICES=0 vllm serve {search_model} --port {port} --dtype auto --max-model-len 40960[/dim]"
                        )
            elif port:
                console.print(
                    f"[green]✓[/green]  Search agent vLLM server is accessible on port [bold]{port}[/bold]"
                )

        # Check browse agent vLLM server if enabled
        use_browse_agent = getattr(cfg, "use_browse_agent", False)
        print(f'use_browse_agent: {use_browse_agent}')
        if use_browse_agent:
            browse_base_url = getattr(cfg, "browse_agent_base_url", None)
            if browse_base_url:
                port = extract_port_from_url(browse_base_url)
                if port and not check_port(port):
                    console.print(
                        f"[yellow]⚠[/yellow]  Browse agent vLLM server is not running on port [bold]{port}[/bold]"
                    )
                    browse_model = getattr(cfg, "browse_agent_model_name", None)
                    if browse_model:
                        if Confirm.ask(
                            f"Launch vLLM server for [cyan]{browse_model}[/cyan] on port {port}?"
                        ):
                            process = launch_vllm_server(
                                browse_model, port, gpu_id=1, logger=self.logger
                            )
                            if process:
                                self._launched_processes.append(process)
                                console.print(
                                    f"[green]✓[/green]  vLLM server launched for {browse_model} on port {port}"
                                )
                            else:
                                console.print(
                                    f"[yellow]⚠[/yellow]  Failed to start vLLM server. Manual launch command:"
                                )
                                console.print(
                                    f"   [dim]CUDA_VISIBLE_DEVICES=1 vllm serve {browse_model} --port {port} --dtype auto --max-model-len 40960[/dim]"
                                )
                        else:
                            console.print(f"[blue]💡[/blue]  Manual launch command:")
                            console.print(
                                f"   [dim]CUDA_VISIBLE_DEVICES=1 vllm serve {browse_model} --port {port} --dtype auto --max-model-len 40960[/dim]"
                            )
                elif port:
                    console.print(
                        f"[green]✓[/green]  Browse agent vLLM server is accessible on port [bold]{port}[/bold]"
                    )

        # Check reranker vLLM server if enabled
        use_reranker = getattr(cfg, "use_reranker", False)
        reranker_type = getattr(cfg, "reranker_type", "pointwise")
        # Skip reranker port check for backends that don't need a vLLM
        # server: listwise_openai* (use Tencent data_eval) and pointwise
        # monoT5/RankT5 (HF in-process inside the MCP server).
        _no_vllm_reranker_types = (
            "listwise_openai",
            "listwise_openai_rubrics",
            "pointwise_monot5",
            "pointwise_rankt5",
        )
        if use_reranker and reranker_type not in _no_vllm_reranker_types:
            reranker_api_url = getattr(cfg, "reranker_api_url", None)
            if reranker_api_url:
                port = extract_port_from_url(reranker_api_url)
                reranker_model = getattr(cfg, "reranker_model_name", None)
                reranker_type = getattr(cfg, "reranker_type", "pointwise")
                if port and not check_port(port):
                    console.print(
                        f"[yellow]⚠[/yellow]  Reranker ({reranker_type}) vLLM server is not running on port [bold]{port}[/bold]"
                    )
                    if reranker_model:
                        console.print(f"[blue]💡[/blue]  Manual launch command:")
                        console.print(
                            f"   [dim]CUDA_VISIBLE_DEVICES=<GPU_ID> vllm serve {reranker_model} --port {port} --dtype auto --max-model-len 40960[/dim]"
                        )
                    console.print(
                        "[red]✗[/red]  Reranker vLLM server is required when use_reranker=true",
                        style="bold red",
                    )
                    raise RuntimeError(
                        f"Reranker vLLM server is not running on port {port}. "
                        f"Please launch it: CUDA_VISIBLE_DEVICES=<GPU_ID> vllm serve {reranker_model} --port {port} --dtype auto --max-model-len 40960"
                    )
                elif port:
                    console.print(
                        f"[green]✓[/green]  Reranker ({reranker_type}) vLLM server is accessible on port [bold]{port}[/bold]"
                    )
            else:
                console.print(
                    "[red]✗[/red]  reranker_api_url is required when use_reranker=true",
                    style="bold red",
                )
                raise RuntimeError(
                    "reranker_api_url must be set when use_reranker=true"
                )

        console.print()
        console.print(Panel.fit("✅ Service Check Complete", style="bold green"))
        console.print()

    def setup_components(
        self,
        mcp_transport_type: Optional[str] = "StreamableHttpTransport",
        mcp_executable: Optional[str] = None,
        mcp_port: Optional[int] = 8000,
    ) -> None:
        cfg = self.configuration
        assert cfg is not None
        # print(cfg)

        # Allow configuration overrides for MCP settings
        if getattr(cfg, "mcp_transport_type", None):
            mcp_transport_type = cfg.mcp_transport_type
        if getattr(cfg, "mcp_executable", None):
            mcp_executable = cfg.mcp_executable
        if getattr(cfg, "mcp_port", None) is not None:
            mcp_port = cfg.mcp_port

        # Number of search candidates to fetch from the search API.
        # Always equals number_documents_to_search; legacy override keeps backward compat.
        num_docs = cfg.number_documents_to_search
        if getattr(cfg, "reranker_num_search_results", -1) and cfg.reranker_num_search_results > 0:
            num_docs = cfg.reranker_num_search_results
        # Final number of docs returned to the agent. reranker_top_n is a legacy alias.
        return_topn = cfg.return_topn
        if getattr(cfg, "reranker_top_n", -1) and cfg.reranker_top_n > 0:
            return_topn = cfg.reranker_top_n

        if cfg.search_tool_name == "serper":
            self.search_tool = SerperSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=num_docs,
                timeout=cfg.search_timeout,
                name="snippet_search",  # <- to test this v20250824 model, we need to set the tool name in a hacky way.
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )

            self.search_tool2 = SerperSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=num_docs,
                timeout=cfg.search_timeout,
                name="google_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.search_tool_name == "s2":
            self.search_tool = SemanticScholarSnippetSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=num_docs,
                timeout=cfg.search_timeout,
                name="snippet_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )

            self.search_tool2 = SerperSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=num_docs,
                timeout=cfg.search_timeout,
                name="google_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.search_tool_name == "s2-only":
            self.search_tool = SemanticScholarSnippetSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=num_docs,
                timeout=cfg.search_timeout,
                name="snippet_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )

            self.search_tool2 = SemanticScholarSnippetSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=num_docs,
                timeout=cfg.search_timeout,
                name="google_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        else:
            raise ValueError(f"Invalid search tool name: {cfg.search_tool_name}")

        # Wrap search tools with reranker if enabled
        if cfg.use_reranker:
            self.search_tool, self.search_tool2 = self._wrap_search_tools_with_reranker(
                cfg, mcp_transport_type, mcp_executable, mcp_port, return_topn
            )
        else:
            # No reranker: truncate the raw search output to return_topn via a thin wrapper.
            self.search_tool = _TopNTruncateSearchTool(self.search_tool, return_topn)
            self.search_tool2 = _TopNTruncateSearchTool(self.search_tool2, return_topn)

        if cfg.browse_tool_name == "serper":
            self.browse_tool = SerperBrowseTool(
                tool_parser=cfg.tool_parser,
                max_pages_to_fetch=cfg.browse_max_pages_to_fetch,
                timeout=cfg.browse_timeout,
                name="browse_webpage",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.browse_tool_name == "crawl4ai":
            self.browse_tool = Crawl4AIBrowseTool(
                tool_parser=cfg.tool_parser,
                max_pages_to_fetch=cfg.browse_max_pages_to_fetch,
                timeout=cfg.browse_timeout,
                name="browse_webpage",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
                context_chars=cfg.browse_context_char_length,
                use_docker_version=cfg.crawl4ai_use_docker_version,
                use_ai2_config=cfg.crawl4ai_use_ai2_config,
            )
        elif cfg.browse_tool_name == "jina":
            self.browse_tool = JinaBrowseTool(
                tool_parser=cfg.tool_parser,
                timeout=cfg.browse_timeout,
                name="browse_webpage",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.browse_tool_name is None:
            self.browse_tool = NoBrowseTool(
                tool_parser=cfg.tool_parser,
                name="browse_webpage",
            )
        else:
            raise ValueError(f"Invalid browse tool name: {cfg.browse_tool_name}")
        print("Using browse tool: ", self.browse_tool)

        if cfg.use_browse_agent:
            with LLMToolClient(
                model_name=cfg.browse_agent_model_name,
                tokenizer_name=cfg.browse_agent_tokenizer_name,
                base_url=cfg.browse_agent_base_url,
                api_key=cfg.browse_agent_api_key,
            ) as client:
                self.browse_agent = WebPageReaderAgentV2(client=client).as_tool(
                    max_tokens=cfg.browse_agent_max_tokens,
                    temperature=cfg.browse_agent_temperature,
                )
                self.composed_browse_tool = ChainedTool(
                    [self.browse_tool, self.browse_agent],
                    name="browse_webpage",
                    tool_parser=cfg.tool_parser,
                    output_formatting="last",
                )
        else:
            self.composed_browse_tool = self.browse_tool

        with LLMToolClient(
            model_name=cfg.search_agent_model_name,
            tokenizer_name=cfg.search_agent_tokenizer_name,
            base_url=cfg.search_agent_base_url,
            api_key=cfg.search_agent_api_key,
        ) as client:
            self.search_agent = SearchAgent(
                client=client,
                tools=[self.search_tool, self.search_tool2, self.composed_browse_tool],
                prompt_version=cfg.prompt_version,
            )
            self.answer_agent = AnswerAgent(
                client=client,
                prompt_version=cfg.prompt_version,
            )

    def _wrap_search_tools_with_reranker(
        self, cfg, mcp_transport_type, mcp_executable, mcp_port, return_topn: int
    ):
        """Wrap search_tool and search_tool2 with SearchWithRerankTool."""
        # Create a browse tool for reranker (fetches page content for reranking)
        if cfg.browse_tool_name == "serper":
            reranker_browse = SerperBrowseTool(
                tool_parser=cfg.tool_parser,
                max_pages_to_fetch=cfg.reranker_browse_max_pages,
                timeout=cfg.reranker_browse_timeout,
                name="_reranker_browse",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.browse_tool_name == "crawl4ai":
            reranker_browse = Crawl4AIBrowseTool(
                tool_parser=cfg.tool_parser,
                max_pages_to_fetch=cfg.reranker_browse_max_pages,
                timeout=cfg.reranker_browse_timeout,
                name="_reranker_browse",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
                use_docker_version=cfg.crawl4ai_use_docker_version,
                use_ai2_config=cfg.crawl4ai_use_ai2_config,
            )
        elif cfg.browse_tool_name == "jina":
            reranker_browse = JinaBrowseTool(
                tool_parser=cfg.tool_parser,
                timeout=cfg.reranker_browse_timeout,
                name="_reranker_browse",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        else:
            raise ValueError(
                f"Invalid browse_tool_name: {cfg.browse_tool_name}"
            )

        # Resolve aliases.
        rtype = cfg.reranker_type
        if rtype == "pointwise":
            rtype = "pointwise_bge"
        elif rtype == "listwise":
            rtype = "listwise_vllm_generic"

        # Setwise rerankers don't truncate to top_n.
        is_setwise = rtype in ("setwise_rank4gen", "setwise_setr", "setwise_rubricranker")
        effective_top_n = -1 if is_setwise else return_topn

        # Create the reranker tool
        if rtype == "pointwise_bge":
            reranker = VllmHostedPointwiseRerankerTool(
                model_name=cfg.reranker_model_name,
                api_url=cfg.reranker_api_url,
                top_n=effective_top_n,
                timeout=cfg.reranker_timeout,
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif rtype in ("pointwise_monot5", "pointwise_rankt5"):
            kind = "monot5" if rtype == "pointwise_monot5" else "rankt5"
            reranker = HFPointwiseRerankerTool(
                model_path=cfg.reranker_model_name,
                model_kind=kind,
                device=cfg.reranker_hf_device,
                doc_max_words=cfg.reranker_doc_max_words,
                top_n=effective_top_n,
                timeout=cfg.reranker_timeout,
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif rtype in ("listwise_rankvicuna", "listwise_rankzephyr"):
            kind = "vicuna" if rtype == "listwise_rankvicuna" else "zephyr"
            reranker = VllmListwiseSlidingWindowRerankerTool(
                model_name=cfg.reranker_model_name,
                api_url=cfg.reranker_api_url,
                model_kind=kind,
                window_size=cfg.reranker_window_size,
                step=cfg.reranker_step,
                max_tokens=cfg.reranker_listwise_max_tokens,
                doc_max_words=cfg.reranker_doc_max_words,
                top_n=effective_top_n,
                timeout=cfg.reranker_timeout,
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif rtype == "setwise_rank4gen":
            reranker = VllmSetwiseRank4GenRerankerTool(
                model_name=cfg.reranker_model_name,
                api_url=cfg.reranker_api_url,
                downstream_model=cfg.reranker_rank4gen_downstream_model,
                downstream_desc=cfg.reranker_rank4gen_downstream_desc,
                max_tokens=cfg.reranker_listwise_max_tokens,
                doc_max_words=cfg.reranker_doc_max_words,
                timeout=cfg.reranker_timeout,
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif rtype == "setwise_setr":
            reranker = VllmSetwiseSETRRerankerTool(
                model_name=cfg.reranker_model_name,
                api_url=cfg.reranker_api_url,
                max_tokens=cfg.reranker_listwise_max_tokens,
                doc_max_words=cfg.reranker_doc_max_words,
                timeout=cfg.reranker_timeout,
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif rtype == "setwise_rubricranker":
            reranker = VllmSetwiseRubricRankerTool(
                model_name=cfg.reranker_model_name,
                api_url=cfg.reranker_api_url,
                prompt_variant=cfg.reranker_rubricranker_prompt_variant,
                max_tokens=cfg.reranker_listwise_max_tokens,
                doc_max_words=cfg.reranker_doc_max_words,
                temperature=cfg.reranker_rubricranker_temperature,
                top_p=cfg.reranker_rubricranker_top_p,
                top_k=cfg.reranker_rubricranker_top_k,
                min_p=cfg.reranker_rubricranker_min_p,
                timeout=cfg.reranker_timeout,
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif rtype == "listwise_singlepass":
            reranker = VllmListwiseSinglePassRerankerTool(
                model_name=cfg.reranker_model_name,
                api_url=cfg.reranker_api_url,
                top_n=effective_top_n,
                max_tokens=cfg.reranker_listwise_max_tokens,
                doc_max_words=cfg.reranker_doc_max_words,
                temperature=cfg.reranker_singlepass_temperature,
                top_p=cfg.reranker_singlepass_top_p,
                top_k=cfg.reranker_singlepass_top_k,
                min_p=cfg.reranker_singlepass_min_p,
                timeout=cfg.reranker_timeout,
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif rtype == "listwise_vllm_generic":
            reranker = VllmHostedListwiseRerankerTool(
                model_name=cfg.reranker_model_name,
                api_url=cfg.reranker_api_url,
                top_n=effective_top_n,
                max_tokens=cfg.reranker_listwise_max_tokens,
                timeout=cfg.reranker_timeout,
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        else:
            raise ValueError(f"Invalid reranker_type: {cfg.reranker_type}")

        if rtype == "listwise_openai":
            print(
                f"Using reranker: {cfg.reranker_type} ({cfg.reranker_openai_model}, "
                f"web_search={cfg.reranker_openai_web_search})"
            )
        elif rtype == "listwise_openai_rubrics":
            print(
                f"Using reranker: {cfg.reranker_type} "
                f"({cfg.reranker_openai_rubrics_model}, rubrics-guided 3-step pipeline)"
            )
        else:
            print(f"Using reranker: {cfg.reranker_type} ({cfg.reranker_model_name})")

        # SearchWithRerankTool's internal top_n cap. For setwise we let it pass through.
        wrapper_top_n = 10**6 if is_setwise else return_topn

        # Wrap both search tools
        wrapped_search_tool = SearchWithRerankTool(
            search_tool=self.search_tool,
            browse_tool=reranker_browse,
            reranker_tool=reranker,
            reranker_top_n=wrapper_top_n,
            doc_max_words=cfg.reranker_doc_max_words,
            skip_browse=cfg.reranker_skip_browse,
            append_page_content_to_snippet=cfg.reranker_append_page_content_to_snippet,
            name=self.search_tool.name,  # Keep original tool name
            tool_parser=cfg.tool_parser,
        )

        wrapped_search_tool2 = SearchWithRerankTool(
            search_tool=self.search_tool2,
            browse_tool=reranker_browse,
            reranker_tool=reranker,
            reranker_top_n=wrapper_top_n,
            doc_max_words=cfg.reranker_doc_max_words,
            skip_browse=cfg.reranker_skip_browse,
            append_page_content_to_snippet=cfg.reranker_append_page_content_to_snippet,
            name=self.search_tool2.name,  # Keep original tool name
            tool_parser=cfg.tool_parser,
        )

        return wrapped_search_tool, wrapped_search_tool2

    async def __call__(
        self,
        problem: str,
        dataset_name: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        verbose: bool = True,
        search_callback: Optional[Any] = None,
        step_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        cfg = self.configuration
        assert cfg is not None

        # Extract history and problem from messages if provided
        history = []
        if messages:
            # Find the last user message as the problem
            last_user_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    last_user_idx = i
                    break

            if last_user_idx != -1:
                problem = messages[last_user_idx]["content"]
                history = messages[:last_user_idx]
            else:
                # Fallback if no user message found (shouldn't happen ideally)
                history = messages

        # import litellm
        # litellm._turn_on_debug()

        # Set the question for the browse agent
        # TODO: This is a bit hectic and hacky, but it works for now
        # The problem: it uses a bad way to enable the runtime dynamics
        if isinstance(self.composed_browse_tool, ChainedTool):
            browse_tool = self.composed_browse_tool.tools[0]
            browse_tool.bm25_query = problem
            browse_agent = self.composed_browse_tool.tools[-1]
            browse_agent.agent.question = problem
        else:
            browse_tool = self.composed_browse_tool
            browse_tool.bm25_query = problem

        results = await self.search_agent(
            question=problem,
            dataset_name=dataset_name,
            history=history,
            max_tokens=cfg.search_agent_max_tokens,
            temperature=cfg.search_agent_temperature,
            max_tool_calls=cfg.search_agent_max_tool_calls,
            verbose=verbose,
            on_step_callback=step_callback,
        )

        if search_callback:
            if asyncio.iscoroutinefunction(search_callback):
                await search_callback(results)
            else:
                search_callback(results)

        browsed_links = []
        searched_links = []
        total_tool_calls = 0
        failed_tool_calls = 0
        failed_tool_call_errors = []
        for tool_output in results.tool_calls:
            total_tool_calls += 1
            if tool_output.error != "":
                failed_tool_calls += 1
                failed_tool_call_errors.append(tool_output.error)

            if tool_output.tool_name in ["snippet_search", "google_search"]:
                # The underlying search tool may, in some error/timeout paths,
                # return a base ``ToolOutput`` (without ``documents``) rather than
                # a ``DocumentToolOutput``. Guard against that to avoid AttributeError.
                if hasattr(tool_output, "documents") and tool_output.documents:
                    searched_links.extend(
                        [document.url for document in tool_output.documents if document.url]
                    )
                else:
                    if tool_output.error:
                        print(
                            f"Warning: {tool_output.tool_name} produced no documents "
                            f"(error={tool_output.error!r}); skipping."
                        )

            if tool_output.tool_name == "browse_webpage":
                if isinstance(self.composed_browse_tool, ChainedTool):
                    if tool_output.raw_output is None:
                        continue
                    if chained_tool_outputs := tool_output.raw_output.get(
                        "tool_outputs"
                    ):
                        for document in chained_tool_outputs[0].documents:
                            if document.url:
                                browsed_links.append(document.url)
                else:
                    if hasattr(tool_output, "documents"):
                        for document in tool_output.documents:
                            if document.url:
                                browsed_links.append(document.url)
                    else:
                        print(
                            f"Warning: browse_webpage tool output has no documents: {tool_output}"
                        )

        browsed_links = list(set(browsed_links))
        searched_links = list(set(searched_links))

        if "<answer>" in results.generated_text:
            return {
                "final_response": self.search_agent.postprocess_output(results),
                "full_traces": results,
                "browsed_links": browsed_links,
                "searched_links": searched_links,
                "total_tool_calls": total_tool_calls,
                "total_failed_tool_calls": failed_tool_calls,
                "failed_tool_call_errors": failed_tool_call_errors,
            }

        answer = await self.answer_agent(
            question=problem,
            history=results.generated_text,
            dataset_name=dataset_name,
            additional_instructions="Now please generate an based on the search results by far:",
            generation_prefix="<answer>",
            max_tokens=cfg.search_agent_max_tokens,
            temperature=cfg.search_agent_temperature,
            verbose=verbose,
            on_step_callback=step_callback,
        )

        if verbose:
            print(results)  # noqa: T201

        answer.tool_calls = [results.model_dump()]

        return {
            "final_response": self.answer_agent.postprocess_output(answer),
            "full_traces": answer,
            "browsed_links": browsed_links,
            "searched_links": searched_links,
        }


if __name__ == "__main__":
    AutoReasonSearchWorkflow.app()
