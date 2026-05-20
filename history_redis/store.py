from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_config

# 加载配置
_config = load_config()

# 无显式 URL 时的默认本地连接
DEFAULT_REDIS_URL = _config.redis.get_connection_url()
SESSION_KEY_MESSAGES = "ctx:{tid}:messages"
SESSION_KEY_SUMMARY = "ctx:{tid}:summary"
DEFAULT_TTL_SECONDS = _config.redis.session_ttl
MAX_LONG_SUMMARY_CHARS = _config.redis.max_summary_chars

_override_client: Any = None


def use_redis_client(client: Any) -> None:
    """测试时注入 fakeredis.FakeRedis 等。"""
    global _override_client
    _override_client = client


def reset_redis_client() -> None:
    global _override_client
    _override_client = None


def _build_redis_url_from_parts() -> str | None:
    """从配置文件构建 Redis URL（兼容方法）。"""
    # 如果配置使用 URL 模式，直接返回
    if _config.redis.connection_mode == "url":
        return _config.redis.url
    
    # 从参数构建
    host = _config.redis.host
    if not host:
        return None
    
    port = _config.redis.port
    db = _config.redis.db
    password = _config.redis.password
    username = _config.redis.username

    def enc(s: str) -> str:
        from urllib.parse import quote
        return quote(str(s), safe="")

    if username and password:
        auth = f"{enc(username)}:{enc(password)}@"
    elif password:
        auth = f":{enc(password)}@"
    else:
        auth = ""
    return f"redis://{auth}{host}:{port}/{db}"


def get_redis_url() -> str:
    """解析本模块使用的 Redis URL（从配置文件读取）。"""
    # 优先使用配置中的 URL
    if _config.redis.connection_mode == "url" and _config.redis.url:
        return _config.redis.url
    
    # 从配置参数构建
    built = _build_redis_url_from_parts()
    if built:
        return built
    return DEFAULT_REDIS_URL


def get_redis():
    if _override_client is not None:
        return _override_client
    import redis

    return redis.Redis.from_url(get_redis_url(), decode_responses=True)


def load_session(thread_id: str) -> dict[str, Any]:
    """
    从 Redis 读取短期 messages 与长期 rolling_summary。
    若 key 不存在，返回空 messages 与空 summary。
    """
    if not thread_id or not str(thread_id).strip():
        return {"messages": [], "rolling_summary": ""}

    r = get_redis()
    mk = SESSION_KEY_MESSAGES.format(tid=thread_id)
    sk = SESSION_KEY_SUMMARY.format(tid=thread_id)

    raw = r.get(mk)
    messages: list = []
    if raw:
        try:
            messages = json.loads(raw)
            if not isinstance(messages, list):
                messages = []
        except json.JSONDecodeError:
            messages = []

    summary = r.get(sk) or ""
    return {"messages": messages, "rolling_summary": summary}


def save_session(
    thread_id: str,
    messages: list,
    rolling_summary: str,
    ttl_seconds: int | None = None,
) -> None:
    """写入 messages 与 rolling_summary，并刷新 TTL。"""
    if not thread_id or not str(thread_id).strip():
        return

    r = get_redis()
    ttl = DEFAULT_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    mk = SESSION_KEY_MESSAGES.format(tid=thread_id)
    sk = SESSION_KEY_SUMMARY.format(tid=thread_id)

    pipe = r.pipeline()
    pipe.set(mk, json.dumps(messages, ensure_ascii=False))
    pipe.set(sk, rolling_summary or "")
    pipe.expire(mk, ttl)
    pipe.expire(sk, ttl)
    pipe.execute()


def merge_long_term_summary(
    previous: str,
    user_question: str,
    assistant_reply: str,
    max_chars: int | None = None,
) -> str:
    """
    将本轮问答追加到长期摘要（滑动窗口截断），不调用 LLM，便于调试与低成本持久化。
    """
    cap = max_chars if max_chars is not None else MAX_LONG_SUMMARY_CHARS
    u = (user_question or "").strip()[:2000]
    a = (assistant_reply or "").strip()[:4000]
    block = f"\n---\nQ: {u}\nA: {a}"
    merged = (previous or "").strip() + block
    if len(merged) > cap:
        merged = merged[-cap:]
    return merged
