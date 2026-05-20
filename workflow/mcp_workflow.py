from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_config
from llm.llm_server import llm_server
from rag.file2chunk import extract_json

_config = load_config()

DEFAULT_MCP_ARXIV_URL = _config.get_mcp_arxiv_url()
DEFAULT_DOWNLOAD_DIR = str(_config.paths.get_mcp_downloads_dir())

ARXIV_KEYWORD_SYSTEM = """You convert a user's natural-language request into a concise arXiv search query.

Output STRICT JSON ONLY (no markdown fence, no commentary):
{
  "arxiv_query": "<string, suitable for arxiv.Search query parameter>",
  "max_results": <integer 1-20>
}

Rules:
- arxiv_query should be short keywords + optional arXiv fields (e.g. ti:\"title\", au:author).
- If the user already gave an arxiv id like 1706.03762, use id:1706.03762 OR the id alone in arxiv_query.
- max_results default 8 if unsure.
"""

ARXIV_DOWNLOAD_PLAN_SYSTEM = """You decide which arXiv papers to download as PDFs.

You receive:
1) The original user request (may mention how many papers, titles, or arxiv ids).
2) A JSON array "search_hits" with objects containing at least: arxiv_id, title, pdf_url.

Output STRICT JSON ONLY:
{
  "items": [
    {"article_id": "1706.03762"},
    {"article_title": "Exact or distinctive title substring if no id"}
  ],
  "max_downloads": <integer, cap how many PDFs to fetch, >=0>
}

Rules:
- Prefer "article_id" when the hit has arxiv_id. Use "article_title" only when id is missing.
- Respect the user's quantity (e.g. "download 2" -> at most 2 items AND set max_downloads=2).
- If the user did not specify a number, set max_downloads to min(3, number of reasonable matches).
- Do not include items not supported by search_hits unless user explicitly gave an arxiv id in their text.
- If nothing should be downloaded, return "items": [] and max_downloads: 0.
"""


class McpArxivState(TypedDict, total=False):
    user_question: str
    llm: llm_server
    # 是否走「下载」分支（由主图根据路由写入）
    include_download: bool
    mcp_server_url: str
    download_output_dir: str

    arxiv_query_keywords: str
    max_search_results: int
    search_hits: list[dict[str, Any]]

    planned_downloads: list[dict[str, Any]]
    planned_max_downloads: int
    download_results: list[dict[str, Any]]
    mcp_error: str

    # 供主图 LLM 使用的上下文摘要（可写入 rag_result）
    mcp_summary_text: str


def _default_mcp_url() -> str:
    return _config.get_mcp_arxiv_url()


def _unwrap_tool_payload(result: Any) -> Any:
    """解析 FastMCP call_tool 返回的 CallToolResult。"""
    if result is None:
        return None
    if getattr(result, "is_error", False):
        return None
    data = getattr(result, "data", None)
    if data is not None:
        return data
    sc = getattr(result, "structured_content", None)
    if isinstance(sc, dict):
        if "result" in sc:
            return sc["result"]
        return sc
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return None


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


async def _mcp_search_arxiv(
    url: str,
    query: str,
    max_results: int,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    try:
        async with Client(url) as client:
            raw = await client.call_tool(
                "search_arxiv",
                {
                    "query": query,
                    "max_results": min(max(1, max_results), 20),
                    "sort_by": "relevance",
                },
            )
    except ToolError as e:
        return None, str(e)
    except OSError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)

    payload = _unwrap_tool_payload(raw)
    if isinstance(payload, list):
        return payload, None
    return None, "unexpected_tool_payload"


async def _mcp_download_batch(
    url: str,
    output_dir: str,
    items: list[dict[str, Any]],
    max_downloads: int,
) -> list[dict[str, Any]]:
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    os.makedirs(output_dir, exist_ok=True)
    cap = max(0, min(int(max_downloads), 20))
    results: list[dict[str, Any]] = []
    count = 0
    async with Client(url) as client:
        for it in items:
            if count >= cap:
                break
            aid = it.get("article_id")
            title = it.get("article_title")
            if not aid and not title:
                continue
            args: dict[str, Any] = {"output_dir": output_dir}
            if aid:
                args["article_id"] = str(aid).strip()
            else:
                args["article_title"] = str(title).strip()
            try:
                raw = await client.call_tool("download_arxiv_paper", args)
            except ToolError as e:
                results.append({"error": str(e)})
                continue
            except Exception as e:
                results.append({"error": str(e)})
                continue
            out = _unwrap_tool_payload(raw)
            if isinstance(out, dict):
                results.append(out)
                if not out.get("error"):
                    count += 1
            else:
                results.append({"error": "unexpected_download_payload", "raw": str(out)})
    return results


def llm_extract_arxiv_keywords_node(state: McpArxivState) -> dict:
    llm = state["llm"]
    url = (state.get("mcp_server_url") or "").strip() or _default_mcp_url()
    messages = [
        {"role": "system", "content": ARXIV_KEYWORD_SYSTEM},
        {"role": "user", "content": f"User request:\n{state['user_question']}"},
    ]
    raw = llm.reasoning(messages)
    q = state["user_question"].strip()
    n = 8
    try:
        data = extract_json(raw)
        q = str(data.get("arxiv_query") or q).strip()
        n = int(data.get("max_results") or 8)
    except (ValueError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    n = max(1, min(n, 20))
    return {
        "mcp_server_url": url,
        "arxiv_query_keywords": q,
        "max_search_results": n,
    }


def mcp_search_node(state: McpArxivState) -> dict:
    url = state.get("mcp_server_url") or _default_mcp_url()
    query = state.get("arxiv_query_keywords") or ""
    n = int(state.get("max_search_results") or 8)
    hits, err = _run_async(_mcp_search_arxiv(url, query, n))
    if err:
        return {
            "search_hits": [],
            "mcp_error": err,
        }
    return {"search_hits": hits or []}


def route_after_search(state: McpArxivState) -> Literal["plan_download", "finalize"]:
    if state.get("include_download"):
        return "plan_download"
    return "finalize"


def llm_plan_downloads_node(state: McpArxivState) -> dict:
    llm = state["llm"]
    hits = state.get("search_hits") or []
    if not hits:
        return {"planned_downloads": [], "planned_max_downloads": 0}

    user_blob = json.dumps(
        {
            "user_question": state["user_question"],
            "search_hits": hits,
        },
        ensure_ascii=False,
    )
    messages = [
        {"role": "system", "content": ARXIV_DOWNLOAD_PLAN_SYSTEM},
        {"role": "user", "content": user_blob},
    ]
    raw = llm.reasoning(messages)
    items: list[dict[str, Any]] = []
    cap = 0
    try:
        data = extract_json(raw)
        items = data.get("items") or []
        if not isinstance(items, list):
            items = []
        cap = int(data.get("max_downloads", 3))
    except (ValueError, json.JSONDecodeError, TypeError, AttributeError):
        items = []
        cap = 0
    cap = max(0, min(cap, 20))
    return {"planned_downloads": items, "planned_max_downloads": cap}


def mcp_batch_download_node(state: McpArxivState) -> dict:
    url = state.get("mcp_server_url") or _default_mcp_url()
    out_dir = (state.get("download_output_dir") or "").strip() or DEFAULT_DOWNLOAD_DIR
    items = state.get("planned_downloads") or []
    cap = int(state.get("planned_max_downloads") or 0)
    if cap == 0:
        return {"download_results": []}
    results = _run_async(_mcp_download_batch(url, out_dir, items, cap))
    return {"download_results": results}


def finalize_mcp_context_node(state: McpArxivState) -> dict:
    hits = state.get("search_hits") or []
    lines: list[str] = ["### arXiv (MCP tools)"]
    if state.get("mcp_error"):
        lines.append(f"(search warning: {state['mcp_error']})")
    lines.append("**Search results:**")
    for i, h in enumerate(hits, 1):
        title = h.get("title", "")
        aid = h.get("arxiv_id", "")
        pdf = h.get("pdf_url", "")
        lines.append(f"{i}. {title} | arxiv:{aid} | {pdf}")

    dls = state.get("download_results") or []
    if dls:
        lines.append("**Downloaded PDFs:**")
        for d in dls:
            if isinstance(d, dict):
                if d.get("error"):
                    lines.append(f"- error: {d.get('error')}")
                else:
                    lines.append(
                        f"- {d.get('title', '')} ({d.get('arxiv_id', '')}) -> {d.get('saved_path', '')}"
                    )

    return {"mcp_summary_text": "\n".join(lines)}


def build_mcp_arxiv_graph():
    graph = StateGraph(McpArxivState)
    graph.add_node("llm_keywords", llm_extract_arxiv_keywords_node)
    graph.add_node("mcp_search", mcp_search_node)
    graph.add_node("llm_plan_downloads", llm_plan_downloads_node)
    graph.add_node("mcp_download", mcp_batch_download_node)
    graph.add_node("finalize", finalize_mcp_context_node)

    graph.add_edge(START, "llm_keywords")
    graph.add_edge("llm_keywords", "mcp_search")
    graph.add_conditional_edges(
        "mcp_search",
        route_after_search,
        {"plan_download": "llm_plan_downloads", "finalize": "finalize"},
    )
    graph.add_edge("llm_plan_downloads", "mcp_download")
    graph.add_edge("mcp_download", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_mcp_arxiv_branch(
    main_state: dict,
    *,
    include_download: bool,
    mcp_server_url: str | None = None,
    download_output_dir: str | None = None,
) -> dict:
    """
    供主工作流占位节点调用：返回 {"rag_result": str, ...} 形式的增量 state。
    """
    payload: McpArxivState = {
        "user_question": main_state.get("UserQuestion") or main_state.get("user_question") or "",
        "llm": main_state["llm"],
        "include_download": include_download,
    }
    if mcp_server_url:
        payload["mcp_server_url"] = mcp_server_url.strip()
    if download_output_dir:
        payload["download_output_dir"] = download_output_dir.strip()

    g = build_mcp_arxiv_graph()
    out = g.invoke(payload)
    return {"rag_result": out.get("mcp_summary_text") or ""}
