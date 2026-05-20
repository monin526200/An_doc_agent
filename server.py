from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config_loader import load_config, sync_to_environ
from llm.llm_server import llm_server
from rag.database_building import (
    chinese_paper_milvus_uri,
    english_paper_milvus_uri,
    insert_json_into_milvus,
)
from workflow.main_workflow import build_workflow

# 加载配置并同步到环境变量（供尚未迁移的模块使用）
_config = load_config()
sync_to_environ(_config)

ALLOWED_SOURCE_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})


def source_file_dir() -> Path:
    return _config.paths.get_source_file_dir()


def _abs_json_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        p = (p or "").strip()
        if not p:
            continue
        out.append(p if os.path.isabs(p) else str(ROOT / p))
    return out


def _sanitize_invoke_result(result: dict[str, Any]) -> dict[str, Any]:
    """去掉不可 JSON 序列化的 llm 等句柄。"""
    drop = {"llm"}
    return {k: v for k, v in result.items() if k not in drop}


app = FastAPI(
    title="MOFL Agent API",
    description="主工作流 + 源文件上传(rag/data/source_file) + Milvus 入库；外部自备 Redis / MCP",
    version="0.1.0",
)


@app.on_event("startup")
def _startup() -> None:
    # 配置已通过 sync_to_environ() 同步到环境变量
    app.state.workflow = build_workflow()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_config.cors.get_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MainWorkflowRequest(BaseModel):
    thread_id: str = Field(..., description="会话 ID，多轮需固定")
    user_question: str = Field(..., description="用户问题")
    use_api: bool = Field(True, description="LLM 是否走 llm_server API 模式")


class MainWorkflowResponse(BaseModel):
    thread_id: str | None = None
    user_question: str | None = None
    llm_response: str | None = None
    rag_result: str | None = None
    workflow_route: str | None = None
    need_rag: bool | None = None
    query_language: str | None = None
    query_category: list[str] | None = None
    messages: Any | None = None
    rolling_summary: str | None = None


class MilvusInsertRequest(BaseModel):
    json_paths: list[str] = Field(..., description="chunk JSON 路径，可相对项目根")
    text_lg: str = Field(..., description="zh | en，对应中/英向量库")
    uri: str | None = Field(None, description="可选；默认按 text_lg 选择 Milvus 文件")


class MilvusInsertResponse(BaseModel):
    rows_inserted: int
    uri_used: str


class UploadedItem(BaseModel):

    original_filename: str
    saved_filename: str
    path_relative_to_root: str


class SourceUploadResponse(BaseModel):
    saved: list[UploadedItem]
    directory: str

class SourceIngestResponse(BaseModel):
    total_sources: int
    skipped_already_processed: int
    pending: int
    chunked_ok: int
    milvus_rows_inserted: int
    errors: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/workflow/main", response_model=MainWorkflowResponse)
def run_main_workflow(body: MainWorkflowRequest):
    graph = app.state.workflow
    llm = llm_server(use_api=body.use_api)
    state = {
        "thread_id": body.thread_id,
        "UserQuestion": body.user_question,
        "llm": llm,
    }
    try:
        result = graph.invoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    clean = _sanitize_invoke_result(result)
    return MainWorkflowResponse(**{k: clean.get(k) for k in MainWorkflowResponse.model_fields})


@app.post("/api/v1/ingest/milvus", response_model=MilvusInsertResponse)
def ingest_milvus(body: MilvusInsertRequest):
    if body.text_lg not in ("zh", "en"):
        raise HTTPException(status_code=400, detail="text_lg 必须是 'zh' 或 'en'")

    paths = _abs_json_paths(body.json_paths)
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise HTTPException(status_code=400, detail=f"文件不存在: {missing}")

    uri = body.uri
    if not uri:
        uri = chinese_paper_milvus_uri if body.text_lg == "zh" else english_paper_milvus_uri
    if not os.path.isabs(uri):
        uri = str(ROOT / uri)

    try:
        n = insert_json_into_milvus(uri, paths, body.text_lg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return MilvusInsertResponse(rows_inserted=n, uri_used=uri)


@app.post("/api/v1/upload/source", response_model=SourceUploadResponse)
def upload_source_files(files: list[UploadFile] = File(..., description="PDF / DOCX / TXT files")):
    if not files:
        raise HTTPException(status_code=400, detail="没有提供文件")

    dest_dir = source_file_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved: list[UploadedItem] = []
    for upload in files:
        raw_name = upload.filename or ""
        base = os.path.basename(raw_name).strip()
        if not base or base in (".", ".."):
            raise HTTPException(status_code=400, detail=f"文件名无效: {raw_name!r}")

        ext = Path(base).suffix.lower()
        if ext not in ALLOWED_SOURCE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的格式 {ext!r}，允许: {sorted(ALLOWED_SOURCE_EXTENSIONS)}",
            )

        target = dest_dir / base
        try:
            data = upload.file.read()
            if not data:
                raise HTTPException(status_code=400, detail=f"空文件: {base}")
            target.write_bytes(data)
        except HTTPException:
            raise
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"写入失败 {base}: {e}") from e

        try:
            rel = str(target.resolve().relative_to(ROOT))
        except ValueError:
            rel = str(target.resolve())
        saved.append(
            UploadedItem(
                original_filename=raw_name,
                saved_filename=base,
                path_relative_to_root=rel.replace("\\", "/"),
            )
        )

    try:
        dir_rel = str(dest_dir.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        dir_rel = str(dest_dir.resolve())

    return SourceUploadResponse(saved=saved, directory=dir_rel)


@app.post("/api/v1/ingest/source", response_model=SourceIngestResponse)
def ingest_source_files_to_milvus(
    pdf_strategy: str = Query("fast", description="PDF 解析策略: fast | hi_res"),
    skip_milvus: bool = Query(False, description="仅生成分块，跳过 Milvus 入库"),
):
    if pdf_strategy not in ("fast", "hi_res"):
        raise HTTPException(status_code=400, detail="pdf_strategy 必须是 'fast' 或 'hi_res'")

    from ingest_source_to_database import ingest_source_directory

    src = str(source_file_dir())
    processed_pkl = str(source_file_dir() / "Processed_file_name_list.pkl")
    llm = llm_server(use_api=True)
    try:
        stats = ingest_source_directory(
            src,
            llm,
            pdf_strategy=pdf_strategy,
            processed_list_path=processed_pkl,
            skip_milvus=skip_milvus,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return SourceIngestResponse(
        total_sources=stats["total_sources"],
        skipped_already_processed=stats["skipped_already_processed"],
        pending=stats["pending"],
        chunked_ok=stats["chunked_ok"],
        milvus_rows_inserted=stats["milvus_rows_inserted"],
        errors=list(stats.get("errors") or []),
    )


if __name__ == "__main__":
    import uvicorn

    host = "0.0.0.0"
    port = 8000
    uvicorn.run("server:app", host=host, port=port, reload=False)
