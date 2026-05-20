import json

from pymilvus import MilvusClient
from rag.database_building import metric_type, embedding_model_path_en, embedding_model_path_zh
from sentence_transformers import SentenceTransformer, CrossEncoder
from llm.llm_server import llm_server
from rag.file2chunk import extract_json, categories as RAG_CATEGORIES

reranker_dir = 'llm/model/bge-reranker-base'
import numpy as np

query_rewrite_prompt = 'You are a query rewriting assistant for a document retrieval system. Rewrite the user\'s ' \
                       'query to be more complete and explicit for document retrieval. Keep the meaning unchanged. ' \
                       'Return the rewritten query ONLY, as a single line of text. '

query_extension_prompt = 'You are a query expansion assistant. Given a user query, generate 3-5 alternative queries ' \
                         'that have the same intent but use different expressions or synonyms. Return the results as ' \
                         'a JSON array of strings ONLY. Example: ["How to train a transformer model?", "Transformer ' \
                         'training tutorial", "Steps to fine-tune a transformer model"] '

results_compression_prompt = 'You are a context compression assistant for a document retrieval system. You are given a ' \
                             'list of document chunks relevant to a query. Summarize and extract only the most relevant ' \
                             'content from each chunk, maintaining factual accuracy. Return the compressed content as a ' \
                             'JSON array, where each element corresponds to one input chunk. Do NOT add any unrelated ' \
                             'information.\n Input format: ["Chunk 1 text...", "Chunk 2 text..."]\n Output format: [' \
                             '"Compressed text for chunk 1", "Compressed text for chunk 2"] '


WORKFLOW_ROUTE_CHAT = "chat"
WORKFLOW_ROUTE_MILVUS_RAG = "milvus_rag"
WORKFLOW_ROUTE_ARXIV_SEARCH = "arxiv_search"
WORKFLOW_ROUTE_ARXIV_SEARCH_DOWNLOAD = "arxiv_search_download"

WORKFLOW_ROUTES = frozenset({
    WORKFLOW_ROUTE_CHAT,
    WORKFLOW_ROUTE_MILVUS_RAG,
    WORKFLOW_ROUTE_ARXIV_SEARCH,
    WORKFLOW_ROUTE_ARXIV_SEARCH_DOWNLOAD,
})

query_classification_prompt = """
You are a query routing assistant for an agent with:
- A local knowledge base (Milvus RAG, domain-tagged documents).
- MCP tools for arXiv: search papers, download PDFs.

Your task: choose exactly ONE route for the user query.

Routes (field "route"):

1) "chat"
   - Casual chat, math/coding without documents, creative writing, or anything that needs no retrieval.

2) "milvus_rag"
   - User needs answers from the LOCAL indexed knowledge base (policies, internal docs, pre-ingested papers by topic).
   - NOT for live arXiv browsing unless they explicitly want the local corpus.

3) "arxiv_search"
   - User wants to FIND or LIST academic papers on arXiv (titles, authors, links, recent work).
   - Examples: "search arxiv for transformer quantization", "what papers exist on X", "papers by Y on arxiv".

4) "arxiv_search_download"
   - User wants to OBTAIN the PDF: download or save an arXiv paper, or search then download.
   - Examples: "download arxiv 1706.03762", "grab the PDF of the attention paper", "search and download recent CLIP paper".

If unsure between arxiv_search and arxiv_search_download: prefer "arxiv_search_download" only when download/save/PDF is clearly requested.

Available knowledge domains (for milvus_rag only):
{categories}

Language options (for milvus_rag only):
- "zh" : Chinese documents
- "en" : English documents

Output format (STRICT JSON ONLY):

{{
  "route": "chat" | "milvus_rag" | "arxiv_search" | "arxiv_search_download",
  "language": "zh" | "en" | null,
  "category": list[string] | null
}}

Rules:
- For route "chat", "arxiv_search", or "arxiv_search_download": set language and category to null.
- For route "milvus_rag": language and category are REQUIRED; category entries MUST come from: {category_list}.
- Do NOT include explanations or text outside JSON.
"""


def rrf_fusion(rankings: list[list], k: int = 60) -> list[tuple]:
    """
    对多个排序列表进行 RRF 融合
    rankings: [[id1, id2, ...], [id3, id1, ...], ...]
    k: RRF 常数，通常取 60
    score = sum(1 / (k + rank))
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs


def hybrid_search(query_text: str, milvus_uri: str, category: list, topk: int = 6, text_lg: str = 'zh'):
    """
    混合检索：稠密向量 + 稀疏向量 (BM25)，RRF 融合后返回按相关性排序的文档内容列表。
    返回 list[str]，每个元素为文档 chunk 的 content，供下游 rerank 使用。
    """
    client = MilvusClient(milvus_uri)

    if text_lg == 'zh':
        embedding_model = SentenceTransformer(embedding_model_path_zh)
    elif text_lg == 'en':
        embedding_model = SentenceTransformer(embedding_model_path_en)
    else:
        raise RuntimeError('语言类型错误')

    query_vec = embedding_model.encode(query_text).tolist()

    filter_expr = " OR ".join(
        [f'ARRAY_CONTAINS(category, "{c}")' for c in category]
    )

    dense_result = client.search(
        collection_name='paper_data',
        data=[query_vec],
        anns_field='embedding',
        search_params={'metric_type': 'IP'},
        limit=topk,
        filter=filter_expr,
        output_fields=['id', 'doc_name', 'content']
    )
    if dense_result and dense_result[0]:
        id_to_content = {}
        vector_ranking = []
        for hit in dense_result[0]:
            entity = hit.get("entity", {})
            doc_id = hit.get("id") or entity.get("id")
            content = entity.get("content")
            if doc_id is not None and content:
                id_to_content[doc_id] = content
                vector_ranking.append(doc_id)
    else:
        raise RuntimeError('稠密向量检索失败')

    sparse_result = client.search(
        collection_name='paper_data',
        data=[query_text],
        anns_field='sparse_vec',
        search_params={'metric_type': 'BM25'},
        limit=topk,
        filter=filter_expr,
        output_fields=['id', 'doc_name', 'content']
    )

    if sparse_result and sparse_result[0]:
        bm25_ranking = []
        for hit in sparse_result[0]:
            entity = hit.get("entity", {})
            doc_id = hit.get("id") or entity.get("id")
            content = entity.get("content")
            if doc_id is not None and content:
                id_to_content.setdefault(doc_id, content)
                bm25_ranking.append(doc_id)
    else:
        raise RuntimeError('稀疏向量检索失败')

    fused = rrf_fusion([vector_ranking, bm25_ranking], k=topk)
    # 将融合后的 (doc_id, score) 映射回 content，返回内容列表供 rerank 使用
    result_contents = [
        id_to_content[doc_id]
        for doc_id, _ in fused[:topk]
        if doc_id in id_to_content
    ]
    return result_contents


def query_classification(query: str, llm: llm_server):
    prompt = f'Query to be categorized: {query}'
    cat_list = ", ".join(RAG_CATEGORIES)
    response = llm.reasoning_wo_messages(
        user_prompt=prompt,
        system_prompt=query_classification_prompt.format(
            categories=cat_list,
            category_list=cat_list,
        ),
    )
    try:
        result = extract_json(response)
    except json.JSONDecodeError:
        raise ValueError("LLM output is not valid JSON")

    route = result.get("route")
    if route not in WORKFLOW_ROUTES:
        # 兼容旧版 JSON（仅 need_rag）
        legacy = result.get("need_rag")
        if legacy is True:
            route = WORKFLOW_ROUTE_MILVUS_RAG
        elif legacy is False:
            route = WORKFLOW_ROUTE_CHAT
        else:
            raise ValueError("Missing or invalid field: route")

    if route == WORKFLOW_ROUTE_CHAT:
        return {
            "workflow_route": route,
            "need_rag": False,
            "language": None,
            "category": None,
        }

    if route == WORKFLOW_ROUTE_ARXIV_SEARCH or route == WORKFLOW_ROUTE_ARXIV_SEARCH_DOWNLOAD:
        return {
            "workflow_route": route,
            "need_rag": False,
            "language": None,
            "category": None,
        }

    # milvus_rag
    language = result.get("language")
    category = result.get("category")

    if language not in ["zh", "en"]:
        raise ValueError("Invalid language")

    if category is None:
        raise ValueError("Missing category")

    return {
        "workflow_route": route,
        "need_rag": True,
        "language": language,
        "category": category,
    }


def rerank_result(query: str, retrieval_results: list[str], topk: int = 4):
    reranker = CrossEncoder(reranker_dir)
   
    pairs = [(query, doc) for doc in retrieval_results]

    # 计算相关性分数
    scores = reranker.predict(pairs)

    # 按分数排序
    ranked = sorted(
        zip(retrieval_results, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[: min(topk, len(ranked))] if ranked else []


def query_rewrite(query: str, llm: llm_server):
    prompt = f'Query to be rewritten: {query}'
    result = llm.reasoning_wo_messages(user_prompt=prompt, system_prompt=query_rewrite_prompt)
    
    return result


def query_extension(query: str, llm: llm_server):
    prompt = f'Query to be expanded: {query}'
    response = llm.reasoning_wo_messages(user_prompt=prompt, system_prompt=query_extension_prompt)
    result = json.loads(response)
    query = " \n ".join(r for r in result)

    return query


def result_compression(query: str, retrieval_results, llm):
    prompt = f'User\'s query: {query} \n Document chunks: {retrieval_results}'
    response = llm.reasoning_wo_messages(user_prompt=prompt, system_prompt=results_compression_prompt)
    result = json.loads(response)
    return result
