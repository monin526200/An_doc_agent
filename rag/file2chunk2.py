"""
数据处理模块 v2：多格式解析、多模态元数据提取、语义分块与分层 chunk 结构。

- PDF / Word：unstructured 分区（PDF 可选 fast / hi_res）
- 文本、表格、图片：结构化抽取；表格进入独立 chunk；图片写入 assets 与可选占位 chunk
- 语义分块：优先使用 LangChain SemanticChunker + 本地 BGE；不可用时回退为段落级向量相似度合并
- 领域与语言：复用 file2chunk 的 LLM 分类（Qwen 等）
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text

from rag.file2chunk import detect_category_lg, save_dir_ch, save_dir_en

_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_RAG_DIR)

EMBED_EN = os.path.join(_PROJECT_ROOT, "llm/model/bge-small-en-v1.5")
EMBED_ZH = os.path.join(_PROJECT_ROOT, "llm/model/bge-small-zh-v1.5")

MIN_SEMANTIC_CHARS = 120
MAX_SEMANTIC_CHUNK = 2500
SIMILARITY_MERGE_THRESHOLD = 0.72


def _guess_doc_language(text: str) -> str:
    if not text or not text.strip():
        return "en"
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return "zh" if cjk > max(8, len(text) * 0.12) else "en"


def _embedding_dir_for_language(lg: str) -> str:
    return EMBED_ZH if lg == "zh" else EMBED_EN


def partition_document(filepath: str, pdf_strategy: str = "fast") -> list:
    """
    按扩展名选择 unstructured 分区入口。
    pdf_strategy: fast | hi_res（hi_res 更利于版式与图片块，但更慢）
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return partition_pdf(filename=str(filepath), strategy=pdf_strategy)
    if ext == ".docx":
        return partition_docx(filename=str(filepath))
    if ext == ".txt":
        with open(filepath, encoding="utf-8") as f:
            return partition_text(text=f.read())
    raise ValueError(f"不支持的文件类型: {ext}（支持 .pdf / .docx / .txt）")


def _element_kind(el: Any) -> str:
    return type(el).__name__


def _element_text(el: Any) -> str:
    t = getattr(el, "text", None) or ""
    return t.strip()


def extract_multimodal_bundle(elements: list) -> dict[str, Any]:
    """
    从 unstructured elements 提取线性正文、表格与图片元数据（多模态管线）。
    """
    narrative_parts: list[str] = []
    tables: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    order = 0

    for el in elements:
        kind = _element_kind(el)
        order += 1
        if kind == "Table":
            text = _element_text(el)
            html = None
            md = getattr(el, "metadata", None)
            if md is not None:
                html = getattr(md, "text_as_html", None)
            block = f"[Table]\n{text}" if text else "[Table]\n(empty)"
            if html:
                block += "\n\n<!-- text_as_html present -->"
            tables.append({"order": order, "text": text, "text_as_html": html})
        elif kind == "Image":
            ref: dict[str, Any] = {"order": order, "type": "image"}
            md = getattr(el, "metadata", None)
            if md is not None:
                d = md.to_dict() if hasattr(md, "to_dict") else {}
                for key in ("image_url", "filename", "image_path", "file_directory"):
                    if d.get(key):
                        ref[key] = d[key]
            images.append(ref)
        else:
            text = _element_text(el)
            if not text:
                continue
            narrative_parts.append(text)

    body_text = "\n\n".join(narrative_parts)
    return {
        "body_text": body_text,
        "tables": tables,
        "images": images,
    }


def _semantic_split_lc(text: str, language_hint: str) -> list[str]:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        return []

    path = _embedding_dir_for_language(language_hint)
    if not os.path.isdir(path):
        return []

    embeddings = HuggingFaceEmbeddings(
        model_name=path,
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=92,
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c and c.strip()]


def _semantic_split_numpy(text: str, language_hint: str) -> list[str]:
    path = _embedding_dir_for_language(language_hint)
    if not os.path.isdir(path):
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paras:
        return []
    if len(paras) == 1:
        return paras

    model = SentenceTransformer(path, trust_remote_code=True)
    emb = model.encode(paras, normalize_embeddings=True)
    merged: list[str] = []
    buf = paras[0]
    for i in range(1, len(paras)):
        sim = float(np.dot(emb[i - 1], emb[i]))
        candidate = buf + "\n\n" + paras[i]
        if sim >= SIMILARITY_MERGE_THRESHOLD and len(candidate) <= MAX_SEMANTIC_CHUNK:
            buf = candidate
        else:
            merged.append(buf)
            buf = paras[i]
    merged.append(buf)
    return merged


def semantic_chunk_body(text: str, language_hint: str | None = None) -> list[str]:
    if not text or not text.strip():
        return []
    lg = language_hint or _guess_doc_language(text)
    if len(text) < MIN_SEMANTIC_CHARS:
        return [text.strip()]

    chunks = _semantic_split_lc(text, lg)
    if not chunks:
        chunks = _semantic_split_numpy(text, lg)
    if not chunks:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = splitter.split_text(text)

    out: list[str] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if len(c) > MAX_SEMANTIC_CHUNK:
            sub = RecursiveCharacterTextSplitter(
                chunk_size=MAX_SEMANTIC_CHUNK,
                chunk_overlap=80,
                length_function=len,
            ).split_text(c)
            out.extend(s.strip() for s in sub if s.strip())
        else:
            out.append(c)
    return out


def build_layered_chunks(
    doc_stem: str,
    semantic_texts: list[str],
    tables: list[dict[str, Any]],
    images: list[dict[str, Any]],
    category: list[str],
    lg: str,
) -> list[dict[str, Any]]:
    """
    分层结构：level=1 语义正文块；level=2 表格块；图片写入 assets，并可选生成 image_ref 块。
    parent_index：指向同文档内上一级语义块索引（表格/图注类块挂在最近语义块上）。
    """
    chunks: list[dict[str, Any]] = []
    parent_for_table: int | None = None

    for i, content in enumerate(semantic_texts):
        parent_for_table = i
        chunks.append(
            {
                "doc_name": doc_stem,
                "content": content,
                "category": category,
                "lg": lg,
                "chunk_type": "semantic_text",
                "hierarchy": {
                    "level": 1,
                    "order": i,
                    "parent_index": None,
                    "role": "body",
                },
            }
        )

    base_order = len(chunks)
    for j, tb in enumerate(tables):
        ttext = (tb.get("text") or "").strip()
        content = f"[Table]\n{ttext}" if ttext else "[Table]\n"
        chunks.append(
            {
                "doc_name": doc_stem,
                "content": content,
                "category": category,
                "lg": lg,
                "chunk_type": "table",
                "hierarchy": {
                    "level": 2,
                    "order": base_order + j,
                    "parent_index": parent_for_table,
                    "role": "table",
                },
            }
        )

    assets = {
        "images": images,
        "tables_meta": [{"order": t.get("order"), "has_html": bool(t.get("text_as_html"))} for t in tables],
    }
    if images:
        summary_bits = []
        for img in images:
            parts = [f"order={img.get('order')}"]
            for k in ("image_url", "filename", "image_path"):
                if img.get(k):
                    parts.append(f"{k}={img[k]}")
            summary_bits.append(" ".join(parts))
        placeholder = "[Image assets]\n" + "\n".join(summary_bits)
        chunks.append(
            {
                "doc_name": doc_stem,
                "content": placeholder[:65000],
                "category": category,
                "lg": lg,
                "chunk_type": "image_ref",
                "hierarchy": {
                    "level": 2,
                    "order": base_order + len(tables),
                    "parent_index": parent_for_table,
                    "role": "image_index",
                },
            }
        )

    if chunks:
        chunks[0]["assets"] = assets
    return chunks


def process_file_to_chunks(
    filepath: str,
    llm,
    pdf_strategy: str = "fast",
    language_hint: str | None = None,
) -> list[dict[str, Any]]:
    """
    单文件：分区 -> 多模态抽取 -> 语义分块 -> LLM 分类 -> 分层 chunk 列表。
    """
    filename = os.path.basename(filepath)
    doc_stem = os.path.splitext(filename)[0]
    elements = partition_document(filepath, pdf_strategy=pdf_strategy)
    bundle = extract_multimodal_bundle(elements)

    body = bundle["body_text"]
    pre_lg = language_hint or _guess_doc_language(body)
    semantic_texts = semantic_chunk_body(body, language_hint=pre_lg)

    if not semantic_texts:
        if bundle["tables"]:
            semantic_texts = ["(No narrative text extracted; see table chunks.)"]
        elif bundle["images"]:
            semantic_texts = ["(No narrative text extracted; see image metadata.)"]
        else:
            semantic_texts = ["(Empty document.)"]

    lg, category = detect_category_lg(semantic_texts if semantic_texts else [body[:8000]], llm)

    layered = build_layered_chunks(
        doc_stem=doc_stem,
        semantic_texts=semantic_texts,
        tables=bundle["tables"],
        images=bundle["images"],
        category=category,
        lg=lg,
    )
    return layered


def file_to_doc_chunks_v2(
    file_dir: str,
    llm,
    pdf_strategy: str = "fast",
    out_dir_en: str | None = None,
    out_dir_ch: str | None = None,
) -> None:
    """
    批量处理目录或单文件，写出与 file2chunk 相同目录约定的 JSON（列表格式，兼容 database_building）。
    """
    out_en = out_dir_en or save_dir_en
    out_ch = out_dir_ch or save_dir_ch
    os.makedirs(out_en, exist_ok=True)
    os.makedirs(out_ch, exist_ok=True)

    paths: list[str] = []
    if os.path.isfile(file_dir):
        paths = [file_dir]
    else:
        for name in os.listdir(file_dir):
            fp = os.path.join(file_dir, name)
            if not os.path.isfile(fp):
                continue
            low = name.lower()
            if low.endswith((".pdf", ".docx", ".txt")):
                paths.append(fp)

    for filepath in paths:
        layered = process_file_to_chunks(filepath, llm, pdf_strategy=pdf_strategy)
        if not layered:
            continue
        lg = layered[0].get("lg", "en")
        name = os.path.splitext(os.path.basename(filepath))[0] + ".json"
        target_dir = out_ch if lg == "ch" else out_en
        os.makedirs(target_dir, exist_ok=True)
        out_path = os.path.join(target_dir, name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(layered, f, ensure_ascii=False, indent=2)


__all__ = [
    "partition_document",
    "extract_multimodal_bundle",
    "semantic_chunk_body",
    "build_layered_chunks",
    "process_file_to_chunks",
    "file_to_doc_chunks_v2",
]
