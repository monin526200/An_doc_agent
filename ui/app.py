"""
Chainlit 前端：调用后端 main_workflow，并提供「源文件入库」「上传到 source_file」操作。

前置：在项目根启动 API，例如：
  uvicorn server:app --host 0.0.0.0 --port 8080

本目录启动 UI（默认连接 http://127.0.0.1:8080）：
  cd ui && chainlit run app.py --port 7860

若 API 地址不同，请修改项目根目录的 config.yaml：
  endpoints:
    api_base_url: "http://host:8080"
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

import chainlit as cl
import requests

# 添加项目根目录到路径以导入 config_loader
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_config

T = TypeVar("T")

# 加载配置
_config = load_config()

API_BASE = _config.get_api_base_url().rstrip("/")

MAIN_WORKFLOW_PATH = "/api/v1/workflow/main"
INGEST_SOURCE_PATH = "/api/v1/ingest/source"
UPLOAD_SOURCE_PATH = "/api/v1/upload/source"

HTTP_TIMEOUT_WORKFLOW = _config.timeouts.workflow
HTTP_TIMEOUT_INGEST = _config.timeouts.ingest
HTTP_TIMEOUT_UPLOAD = _config.timeouts.upload


async def _in_thread(fn: Callable[[], T]) -> T:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn)


def toolbar_actions() -> list[cl.Action]:
    return [
        cl.Action(name="ingest_db", label="同步 source_file 到数据库", payload={}),
        cl.Action(name="upload_files", label="上传文件到 source_file", payload={}),
    ]


def _session_thread_id() -> str:
    tid = cl.user_session.get("thread_id")
    if not tid:
        tid = str(uuid.uuid4())
        cl.user_session.set("thread_id", tid)
    return str(tid)


def _format_workflow_reply(data: dict[str, Any]) -> str:
    parts: list[str] = []
    route = data.get("workflow_route")
    if route:
        parts.append(f"**路由**: `{route}`")
    if data.get("llm_response"):
        parts.append(str(data["llm_response"]))
    if data.get("rag_result"):
        parts.append("\n\n**RAG**\n\n" + str(data["rag_result"]))
    return "\n\n".join(parts).strip() if parts else "(后端未返回正文)"


def _post_upload_from_paths(files: list[Any]) -> requests.Response:
    opened: list[Any] = []
    try:
        multipart: list[tuple[str, Any]] = []
        for f in files:
            path = getattr(f, "path", None)
            name = getattr(f, "name", None) or os.path.basename(path or "") or "upload.bin"
            if not path or not os.path.isfile(path):
                raise ValueError(f"无法读取上传项: {name!r}")
            fh = open(path, "rb")
            opened.append(fh)
            multipart.append(("files", (name, fh, getattr(f, "type", None) or "application/octet-stream")))
        return requests.post(
            f"{API_BASE}{UPLOAD_SOURCE_PATH}",
            files=multipart,
            timeout=HTTP_TIMEOUT_UPLOAD,
        )
    finally:
        for fh in opened:
            fh.close()


async def _run_upload_flow() -> None:
    files = await cl.AskFileMessage(
        content="选择 PDF / DOCX / TXT（可多选），将保存到后端 `source_file` 目录（与 `API_SOURCE_FILE_DIR` 一致）。",
        accept=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            ".pdf",
            ".docx",
            ".txt",
        ],
        max_size_mb=200,
        max_files=10,
    ).send()
    if not files:
        await cl.Message(content="已取消上传。").send()
        return
    try:
        r = await _in_thread(lambda: _post_upload_from_paths(files))
    except Exception as e:
        await cl.Message(content=f"上传请求失败: {e}").send()
        return
    if r.ok:
        try:
            body = r.json()
        except Exception:
            body = {}
        saved = body.get("saved") or []
        names = ", ".join(
            str((s or {}).get("saved_filename") or (s or {}).get("original_filename") or "?")
            for s in saved
        )
        d = body.get("directory", "")
        await cl.Message(content=f"上传成功：{names}\n\n保存目录（相对项目根）: `{d}`").send()
    else:
        detail = r.text
        try:
            err = r.json().get("detail", detail)
            detail = str(err)
        except Exception:
            pass
        await cl.Message(content=f"上传失败 HTTP {r.status_code}: {detail}").send()


@cl.on_chat_start
async def start() -> None:
    await cl.Message(
        content=(
            f"后端: `{API_BASE}`\n\n"
            "在下方输入问题即会通过 **main_workflow** 多轮对话（会话按本页 session 固定 `thread_id`）。\n\n"
            "**按钮**: 将 `source_file` 中**新文件**切块并入库；或上传本地文件到该目录。\n\n"
            "也可在对话中发送 **`/upload`** 触发文件选择。"
        ),
        actions=toolbar_actions(),
    ).send()


@cl.action_callback("ingest_db")
async def on_ingest_db(action: cl.Action) -> None:
    await cl.Message(content="正在将 `source_file` 中的新文档切块并写入 Milvus，请稍候（大文件可能较久）…").send()
    try:
        r = await _in_thread(
            lambda: requests.post(
                f"{API_BASE}{INGEST_SOURCE_PATH}",
                params={"pdf_strategy": "fast", "skip_milvus": False},
                timeout=HTTP_TIMEOUT_INGEST,
            ),
        )
    except requests.RequestException as e:
        await cl.Message(content=f"请求失败: {e}").send()
        return
    if r.ok:
        try:
            s = r.json()
        except Exception:
            s = {}
        err_list = s.get("errors") or []
        lines = [
            "**入库结果**",
            f"- 源文件总数: {s.get('total_sources')}",
            f"- 已跳过（已处理）: {s.get('skipped_already_processed')}",
            f"- 待处理: {s.get('pending')}",
            f"- 本次切块完成: {s.get('chunked_ok')}",
            f"- Milvus 插入行数: {s.get('milvus_rows_inserted')}",
        ]
        if err_list:
            lines.append("- **错误**:")
            lines.extend(f"  - {e}" for e in err_list)
        await cl.Message(content="\n".join(lines)).send()
    else:
        detail = r.text
        try:
            detail = str(r.json().get("detail", detail))
        except Exception:
            pass
        await cl.Message(content=f"入库失败 HTTP {r.status_code}: {detail}").send()


@cl.action_callback("upload_files")
async def on_upload_files(action: cl.Action) -> None:
    await _run_upload_flow()


@cl.on_message
async def main(message: cl.Message) -> None:
    text = (message.content or "").strip()
    if text.lower() in ("/upload", "/上传"):
        await _run_upload_flow()
        return

    if not text:
        await cl.Message(content="请输入问题，或发送 `/upload` 上传文件。").send()
        return

    payload = {
        "thread_id": _session_thread_id(),
        "user_question": text,
        "use_api": True,
    }
    try:
        r = await _in_thread(
            lambda: requests.post(
                f"{API_BASE}{MAIN_WORKFLOW_PATH}",
                json=payload,
                timeout=HTTP_TIMEOUT_WORKFLOW,
            ),
        )
    except requests.RequestException as e:
        await cl.Message(content=f"调用主工作流失败: {e}", actions=toolbar_actions()).send()
        return

    if not r.ok:
        detail = r.text
        try:
            detail = str(r.json().get("detail", detail))
        except Exception:
            pass
        await cl.Message(
            content=f"工作流 HTTP {r.status_code}: {detail}",
            actions=toolbar_actions(),
        ).send()
        return

    try:
        data = r.json()
    except Exception:
        data = {}
    await cl.Message(
        content=_format_workflow_reply(data),
        actions=toolbar_actions(),
    ).send()
