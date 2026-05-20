from __future__ import annotations

import argparse
import json
import sys

from history_redis.store import (
    SESSION_KEY_MESSAGES,
    SESSION_KEY_SUMMARY,
    get_redis,
    get_redis_url,
    load_session,
)


def _print_session(thread_id: str) -> int:
    tid = (thread_id or "").strip()
    if not tid:
        sys.stderr.write("请提供 --thread-id / -t\n")
        return 2

    r = get_redis()
    mk = SESSION_KEY_MESSAGES.format(tid=tid)
    sk = SESSION_KEY_SUMMARY.format(tid=tid)

    data = load_session(tid)
    ttl_m = r.ttl(mk)
    ttl_s = r.ttl(sk)

    print(f"REDIS_URL (解析结果): {get_redis_url()}")
    print(f"thread_id: {tid}")
    print(f"Key messages: {mk}  (TTL 秒, -2=无键, -1=无过期): {ttl_m}")
    print(f"Key summary: {sk}  (TTL 秒): {ttl_s}")
    print()

    messages = data.get("messages") or []
    print(f"短期记忆 messages: 共 {len(messages)} 条")
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print()

    summary = data.get("rolling_summary") or ""
    print(f"长期摘要 rolling_summary: {len(summary)} 字符")
    print(summary if summary else "(空)")
    return 0


def _list_keys(pattern: str = "ctx:*") -> int:
    r = get_redis()
    keys = sorted(r.keys(pattern))
    print(f"REDIS_URL: {get_redis_url()}")
    print(f"匹配 {pattern!r} 的 key 共 {len(keys)} 个：")
    for k in keys:
        t = r.ttl(k)
        print(f"  {k}  TTL={t}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="查看会话在 Redis 中的存储")
    p.add_argument("-t", "--thread-id", default="", help="与 main 里 CHAT_THREAD_ID 一致")
    p.add_argument(
        "--list-keys",
        action="store_true",
        help="列出 ctx:* 键（多会话时便于找到 thread_id）",
    )
    p.add_argument(
        "--pattern",
        default="ctx:*",
        help="与 --list-keys 共用，默认 ctx:*",
    )
    args = p.parse_args(argv)

    if args.list_keys:
        return _list_keys(args.pattern)
    if args.thread_id:
        return _print_session(args.thread_id)
    sys.stderr.write("请指定 -t <thread_id> 或 --list-keys\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
