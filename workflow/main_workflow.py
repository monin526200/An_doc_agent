from copy import deepcopy

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from history_redis.store import load_session, merge_long_term_summary, save_session
from llm.llm_server import llm_server
from rag.rag_query import (
    query_classification,
    WORKFLOW_ROUTE_ARXIV_SEARCH,
    WORKFLOW_ROUTE_ARXIV_SEARCH_DOWNLOAD,
    WORKFLOW_ROUTE_CHAT,
    WORKFLOW_ROUTE_MILVUS_RAG,
)
from workflow.rag_workflow import build_rag_chain
from workflow.mcp_workflow import run_mcp_arxiv_branch

LLM_CHAT_SYSTEM_PROMPT = """
You are a friendly, capable assistant having a natural conversation with the user.

- Respond like a thoughtful human: warm, clear, and easy to follow; match the user's tone when appropriate.
- You may use general knowledge, reasoning, and the ongoing chat (including any long-term memory block below) to help.
- For greetings, small talk, opinions, brainstorming, or vague questions, engage helpfully without demanding document context.
- Be honest about uncertainty; do not invent facts when it matters.
- Stay concise when a short answer suffices; expand when the user clearly wants depth.
""".strip()


LLM_DOCUMENT_SYSTEM_PROMPT = """
You are a helpful document question answering assistant.

Answer the user's question using ONLY the provided context.
Do not use outside knowledge.

If the context does not contain enough information to answer the question, say that the information is not available in the documents.

Guidelines:
- Be concise and factual.
- Quote key information from the context when helpful.
- Do not fabricate information.
""".strip()


LONG_TERM_PREFIX = (
    "\n\n[Long-term conversation memory (rolling summary of past turns)]\n"
)

user_prompt_template = "Context:{context} \n\n Question:{question}"


class State(TypedDict, total=False):
    thread_id: str
    UserQuestion: str
    messages: list
    rolling_summary: str
    llm_response: str
    rag_result: str
    llm: llm_server
    need_rag: bool
    query_language: str
    query_category: list[str]
    # question_classification：chat | milvus_rag | arxiv_search | arxiv_search_download
    workflow_route: str


def load_redis_node(state: State) -> dict:
    tid = state.get("thread_id") or ""
    if not str(tid).strip():
        return {}
    data = load_session(str(tid))
    return {
        "messages": data["messages"],
        "rolling_summary": data.get("rolling_summary") or "",
    }


def save_redis_node(state: State) -> dict:
    tid = state.get("thread_id") or ""
    if not str(tid).strip():
        return {}
    messages = state.get("messages") or []
    prev_summary = state.get("rolling_summary") or ""
    new_summary = merge_long_term_summary(
        prev_summary,
        state.get("UserQuestion", ""),
        state.get("llm_response", ""),
    )
    save_session(str(tid), messages, new_summary)
    return {"rolling_summary": new_summary}


def Question_Classifier_node(state: State):
    cl_result = query_classification(state["UserQuestion"], state["llm"])
    return {
        "workflow_route": cl_result["workflow_route"],
        "need_rag": cl_result["need_rag"],
        "query_language": cl_result["language"],
        "query_category": cl_result["category"],
    }


def route_after_classification(state: State) -> str:
    r = state.get("workflow_route") or WORKFLOW_ROUTE_CHAT
    if r == WORKFLOW_ROUTE_MILVUS_RAG:
        return WORKFLOW_ROUTE_MILVUS_RAG
    if r == WORKFLOW_ROUTE_ARXIV_SEARCH:
        return WORKFLOW_ROUTE_ARXIV_SEARCH
    if r == WORKFLOW_ROUTE_ARXIV_SEARCH_DOWNLOAD:
        return WORKFLOW_ROUTE_ARXIV_SEARCH_DOWNLOAD
    return WORKFLOW_ROUTE_CHAT


def arxiv_search_branch_node(state: State) -> dict:
    return run_mcp_arxiv_branch(state, include_download=False)


def arxiv_search_download_branch_node(state: State) -> dict:
    return run_mcp_arxiv_branch(state, include_download=True)


def LLM_answer_node(state: State):
    messages = deepcopy(state.get("messages") or [])
    summary = (state.get("rolling_summary") or "").strip()
    route = state.get("workflow_route") or WORKFLOW_ROUTE_CHAT
    is_chat = route == WORKFLOW_ROUTE_CHAT

    system_prompt_base = LLM_CHAT_SYSTEM_PROMPT if is_chat else LLM_DOCUMENT_SYSTEM_PROMPT
    system_content = (
        system_prompt_base + LONG_TERM_PREFIX + summary if summary else system_prompt_base
    )

    found_system = False
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            messages[i] = {**m, "content": system_content}
            found_system = True
            break
    if not found_system:
        messages.insert(0, {"role": "system", "content": system_content})

    if is_chat:
        user_content = (state.get("UserQuestion") or "").strip()
    else:
        user_content = user_prompt_template.format(
            question=state["UserQuestion"],
            context=state.get("rag_result") or "",
        )
    messages.append({"role": "user", "content": user_content})
    response = state["llm"].reasoning(messages)
    messages.append({"role": "assistant", "content": response})
    return {"llm_response": response, "messages": messages}


def rag_node(state: State):
    lang = state.get("query_language") or "en"
    category = state.get("query_category") or ["machine_learning"]
    rag_state = {
        "query": state["UserQuestion"],
        "llm": state["llm"],
        "optimized_query": None,
        "search_result": None,
        "reranked_result": None,
        "compressed_result": None,
        "language": lang,
        "category": category,
    }
    chain = build_rag_chain()
    out = chain.invoke(rag_state)
    compressed = out.get("compressed_result")
    if isinstance(compressed, list):
        rag_text = "\n".join(str(x) for x in compressed)
    else:
        rag_text = str(compressed or "")

    return {"rag_result": rag_text}


def build_workflow():
    builder = StateGraph(State)

    builder.add_node("load_redis", load_redis_node)
    builder.add_node("save_redis", save_redis_node)
    builder.add_node("LLM", LLM_answer_node)
    builder.add_node("question_classification", Question_Classifier_node)
    builder.add_node("rag", rag_node)
    builder.add_node("arxiv_search_branch", arxiv_search_branch_node)
    builder.add_node("arxiv_search_download_branch", arxiv_search_download_branch_node)

    builder.add_edge(START, "load_redis")
    builder.add_edge("load_redis", "question_classification")
    builder.add_conditional_edges(
        "question_classification",
        route_after_classification,
        {
            WORKFLOW_ROUTE_CHAT: "LLM",
            WORKFLOW_ROUTE_MILVUS_RAG: "rag",
            WORKFLOW_ROUTE_ARXIV_SEARCH: "arxiv_search_branch",
            WORKFLOW_ROUTE_ARXIV_SEARCH_DOWNLOAD: "arxiv_search_download_branch",
        },
    )
    builder.add_edge("rag", "LLM")
    builder.add_edge("arxiv_search_branch", "LLM")
    builder.add_edge("arxiv_search_download_branch", "LLM")
    builder.add_edge("LLM", "save_redis")
    builder.add_edge("save_redis", END)

    return builder.compile()
