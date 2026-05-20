from typing import TypedDict
from langchain_core.runnables import RunnableLambda
from llm.llm_server import llm_server
from rag.rag_query import query_extension, query_rewrite, result_compression, rerank_result, hybrid_search
from rag.database_building import chinese_paper_milvus_uri, english_paper_milvus_uri


class RagState:
    query: str
    optimized_query: str
    search_result: list[str]
    reranked_result: list[str]
    compressed_result: list[str]
    llm: llm_server
    language: str
    category: list[str]


def query_optimize_node(state: RagState):
    optimized_query = query_rewrite(state['query'], state['llm'])
    expanded_query = query_extension(optimized_query, state['llm'])
    state["optimized_query"] = expanded_query
    return state


def hybrid_search_node(state: RagState):
    if state['language'] == 'en':
        uri = english_paper_milvus_uri
    else:
        uri = chinese_paper_milvus_uri
    search_result = hybrid_search(query_text=state['optimized_query'], milvus_uri=uri, category=state['category'],
                                  topk=10, text_lg=state['language'])
    state["search_result"] = search_result
    return state


def rerank_node(state: RagState):
    ranked_pairs = rerank_result(query=state['query'], retrieval_results=state['search_result'], topk=5)
    state["reranked_result"] = [doc for doc, _ in ranked_pairs]
    return state


def context_compression_node(state: RagState):
    compressed_result = result_compression(query=state['query'], retrieval_results=state['reranked_result'],
                                           llm=state['llm'])
    state["compressed_result"] = compressed_result
    return state


def build_rag_chain():
    query_optimize = RunnableLambda(query_optimize_node)
    search = RunnableLambda(hybrid_search_node)
    rerank = RunnableLambda(rerank_node)
    compress = RunnableLambda(context_compression_node)

    rag_chain = (
            query_optimize
            | search
            | rerank
            | compress
    )

    return rag_chain


if __name__ == '__main__':
    llm = llm_server(use_api=True)
    test_query = 'this is a query for test, search information about cv'
    state = {
        'query': test_query,
        'optimized_query': None,
        'search_result': None,
        'reranked_result': None,
        'compressed_result': None,
        'llm': llm,
        'language': 'en',
        'category': ['computer_vision']
    }
    chain = build_rag_chain()
    result=chain.invoke(state)

    print(result)
