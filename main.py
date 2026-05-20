import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_config
from history_redis.store import load_session
from workflow.main_workflow import build_workflow
from llm.llm_server import llm_server

_config = load_config()

if __name__ == "__main__":

    agent = build_workflow()
    model = llm_server(use_api=True)

    # 多轮对话固定 thread_id，未设置则自动生成
    thread_id = _config.runtime.chat_thread_id or str(uuid.uuid4())

    print(f"thread_id={thread_id}")
    print("  <new> 新会话")
    print("  <end> 退出")
    print("  <memory> 查看记忆\n")
    while True:
        query = input("你说:\n").strip()
        if not query:
            continue
        if query == "<end>":
            print("再见")
            break
        if query == "<new>":
            thread_id = str(uuid.uuid4())
            print(f"新会话: {thread_id}\n")
            continue
        if query == "<memory>":
            snap = load_session(thread_id)
            msgs = snap.get("messages") or []
            summ = snap.get("rolling_summary") or ""
            print(f"[redis] 消息={len(msgs)}, 摘要={len(summ)} 字符\n")
            if summ:
                print(summ[:2000] + ("…\n" if len(summ) > 2000 else "\n"))
            continue

        state = {
            "thread_id": thread_id,
            "UserQuestion": query,
            "llm": model,
        }
        result = agent.invoke(state)
        print(f'回复: {result["llm_response"]}\n 来源:{result["rag_result"]}\n')
