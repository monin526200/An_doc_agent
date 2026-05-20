"""
arXiv MCP Server - 基于 FastMCP 和 FastAPI 的工具调用服务端

提供两个工具:
1. search_arxiv - 根据关键词在 arXiv 上检索相关文章
2. download_arxiv_paper - 根据文章 ID 或名称下载文章
"""

import os
from typing import Any

import arxiv
from fastapi import FastAPI
from fastmcp import FastMCP


def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
) -> list[dict[str, Any]]:
    """
    根据关键词在 arXiv 上检索相关文章。
  
    Args:
        query: 搜索关键词，支持 arXiv 高级查询语法（如 ti:title, au:author）
        max_results: 返回的最大结果数量，默认 10
        sort_by: 排序方式，可选 "relevance"（相关性）、"submittedDate"（提交日期）、"lastUpdatedDate"（更新日期）

    Returns:
        文章列表，每项包含 title、authors、summary、entry_id、pdf_url 等信息
    """
    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
    }
    sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=min(max_results, 50),  # 限制最大 50 条
        sort_by=sort_criterion,
    )

    results = []
    for result in client.results(search):
        results.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "summary": result.summary[:500] + "..." if len(result.summary) > 500 else result.summary,
            "entry_id": result.entry_id,
            "arxiv_id": result.get_short_id(),
            "pdf_url": result.pdf_url,
            "published": str(result.published) if result.published else None,
            "primary_category": result.primary_category,
        })
    return results


def download_arxiv_paper(
    article_id: str | None = None,
    article_title: str | None = None,
    output_dir: str = "./downloads",
) -> dict[str, Any]:
    """
    根据文章 ID 或名称下载 arXiv 文章 PDF。

    可通过 article_id（arXiv ID，如 2301.12345）或 article_title（文章标题）指定要下载的文章。
    若使用 article_title，将搜索并下载第一个匹配结果。

    Args:
        article_id: arXiv 文章 ID，如 "2301.12345" 或 "2301.12345v1"
        article_title: 文章标题（当 article_id 未提供时使用，将搜索并下载第一个匹配）
        output_dir: 下载保存目录，默认 ./downloads

    Returns:
        包含 saved_path、arxiv_id、title 的字典，失败时包含 error 信息
    """
    if not article_id and not article_title:
        return {"error": "必须提供 article_id 或 article_title 之一"}

    client = arxiv.Client()

    if article_id:
        # 直接通过 ID 获取
        search = arxiv.Search(id_list=[article_id.strip()])
        results = list(client.results(search))
        if not results:
            return {"error": f"未找到 ID 为 {article_id} 的文章"}
        result = results[0]
    else:
        # 通过标题搜索
        search = arxiv.Search(
            query=f"ti:{article_title}",
            max_results=5,
        )
        results = list(client.results(search))
        if not results:
            return {"error": f"未找到标题包含 '{article_title}' 的文章"}
        # 取第一个结果（最相关）
        result = results[0]

    os.makedirs(output_dir, exist_ok=True)
    try:
        saved_path = result.download_pdf(dirpath=output_dir)
        return {
            "saved_path": os.path.abspath(saved_path),
            "arxiv_id": result.get_short_id(),
            "title": result.title,
        }
    except Exception as e:
        return {
            "error": str(e),
            "arxiv_id": result.get_short_id(),
            "title": result.title,
        }


mcp = FastMCP(
    "arXiv Tools",
    instructions="arXiv 论文检索与下载工具",
    tools=[search_arxiv, download_arxiv_paper],
)

# 创建 MCP 的 HTTP 应用
mcp_app = mcp.http_app(path="/mcp")

# 创建 FastAPI 应用并挂载 MCP
app = FastAPI(
    title="arXiv MCP Server",
    description="提供 arXiv 论文检索与下载的 MCP 工具服务",
    version="1.0.0",
    lifespan=mcp_app.lifespan,
)

# 挂载 MCP 服务，访问路径为 /arxiv/mcp
app.mount("/arxiv", mcp_app)


@app.get("/")
def root():
    """健康检查与 API 信息"""
    return {
        "service": "arXiv MCP Server",
        "mcp_endpoint": "/arxiv/mcp",
        "tools": ["search_arxiv", "download_arxiv_paper"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
