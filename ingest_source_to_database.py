"""
源文档目录 -> file2chunk2 分层 chunk（JSON）-> Milvus 增量写入。

已处理源文件名保存在 pickle 列表中（默认与 file2chunk 约定的路径一致），
按「原始文件名」跳过已处理文档，避免重复切块与重复入库（每次仅插入新生成的 JSON）。

用法（在项目根目录）：
  python ingest_source_to_database.py /path/to/papers
  python ingest_source_to_database.py /path/to/papers --use-api --pdf-strategy fast
  python ingest_source_to_database.py /path/to/papers --skip-milvus   # 只生成 JSON
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PROCESSED_PKL = os.path.join(ROOT, "rag", "data", "source_file", "Processed_file_name_list.pkl")
DEFAULT_SOURCE_DIR = os.path.join(ROOT, "rag", "data", "source_file")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from llm.llm_server import llm_server
from rag.database_building import (
    chinese_paper_milvus_uri,
    english_paper_milvus_uri,
    insert_json_into_milvus,
)
from rag.file2chunk import save_dir_ch, save_dir_en
from rag.file2chunk2 import process_file_to_chunks


def load_processed_names(pkl_path: str) -> set[str]:
    """已处理列表存为 list[str]，每项为源文件 basename（如 paper.pdf）。"""
    if not os.path.isfile(pkl_path):
        return set()
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except EOFError:
        return set()
    if isinstance(data, (list, tuple, set)):
        return {str(x) for x in data}
    return set()


def save_processed_names(pkl_path: str, names: set[str]) -> None:
    parent = os.path.dirname(os.path.abspath(pkl_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(sorted(names), f)


def iter_source_files(source_dir: str) -> list[str]:
    paths: list[str] = []
    abs_dir = os.path.abspath(source_dir)
    if os.path.isfile(abs_dir):
        low = abs_dir.lower()
        if low.endswith((".pdf", ".docx", ".txt")):
            paths.append(abs_dir)
        return paths
    if not os.path.isdir(abs_dir):
        raise FileNotFoundError(f"源路径不存在: {source_dir}")
    for name in sorted(os.listdir(abs_dir)):
        fp = os.path.join(abs_dir, name)
        if not os.path.isfile(fp):
            continue
        low = name.lower()
        if low.endswith((".pdf", ".docx", ".txt")):
            paths.append(fp)
    return paths


def ingest_source_directory(
    source_dir: str,
    llm: llm_server,
    *,
    pdf_strategy: str = "fast",
    processed_list_path: str = DEFAULT_PROCESSED_PKL,
    skip_milvus: bool = False,
) -> dict:
    processed = load_processed_names(processed_list_path)
    all_sources = iter_source_files(source_dir)
    pending = [p for p in all_sources if os.path.basename(p) not in processed]

    stats: dict = {
        "total_sources": len(all_sources),
        "skipped_already_processed": len(all_sources) - len(pending),
        "pending": len(pending),
        "chunked_ok": 0,
        "milvus_rows_inserted": 0,
        "errors": [],
    }

    os.makedirs(save_dir_en, exist_ok=True)
    os.makedirs(save_dir_ch, exist_ok=True)

    for filepath in pending:
        basename = os.path.basename(filepath)
        try:
            # 上次若已写出 JSON 但入库失败，仅补插 Milvus，避免重复切块/重复向量
            stem = os.path.splitext(basename)[0]
            json_ch = os.path.join(save_dir_ch, stem + ".json")
            json_en = os.path.join(save_dir_en, stem + ".json")
            orphan_json: str | None = None
            orphan_lg: str | None = None
            if os.path.isfile(json_ch):
                orphan_json, orphan_lg = json_ch, "zh"
            elif os.path.isfile(json_en):
                orphan_json, orphan_lg = json_en, "en"

            if orphan_json and orphan_lg:
                rows = 0
                if not skip_milvus:
                    uri = (
                        chinese_paper_milvus_uri
                        if orphan_lg == "zh"
                        else english_paper_milvus_uri
                    )
                    rows = insert_json_into_milvus(uri, [orphan_json], orphan_lg)
                    stats["milvus_rows_inserted"] += rows
                processed.add(basename)
                save_processed_names(processed_list_path, processed)
                stats["chunked_ok"] += 1
                continue

            layered = process_file_to_chunks(filepath, llm, pdf_strategy=pdf_strategy)
            if not layered:
                stats["errors"].append(f"{basename}: empty chunks")
                continue

            lg = layered[0].get("lg", "en")
            json_name = os.path.splitext(basename)[0] + ".json"
            target_dir = save_dir_ch if lg == "ch" else save_dir_en
            os.makedirs(target_dir, exist_ok=True)
            out_path = os.path.join(target_dir, json_name)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(layered, f, ensure_ascii=False, indent=2)

            rows = 0
            if not skip_milvus:
                milvus_lg = "zh" if lg == "ch" else "en"
                uri = chinese_paper_milvus_uri if milvus_lg == "zh" else english_paper_milvus_uri
                rows = insert_json_into_milvus(uri, [out_path], milvus_lg)
                stats["milvus_rows_inserted"] += rows

            processed.add(basename)
            save_processed_names(processed_list_path, processed)
            stats["chunked_ok"] += 1
        except Exception as e:
            stats["errors"].append(f"{basename}: {e}")

    return stats


def main(argv: list[str] | None = None) -> int:
    os.chdir(ROOT)
    p = argparse.ArgumentParser(
        description="将源目录中的新文档切块写入 rag/data/sliced_doc_* 并增量插入 Milvus",
    )
    p.add_argument("--source_dir", default=DEFAULT_SOURCE_DIR, help="源 PDF/Docx/Txt 所在目录，或单个文件路径")
    p.add_argument(
        "--use-api",
        choices=("true", "false"),
        default="true",
        help="是否使用 llm_server API 模式做领域/语言分类（true/false）",
    )
    p.add_argument("--pdf-strategy", default="fast", choices=("fast", "hi_res"))
    p.add_argument(
        "--processed-list",
        default=DEFAULT_PROCESSED_PKL,
        help="已处理源文件 basename 列表（pickle list）路径",
    )
    p.add_argument("--skip-milvus", action="store_true", help="只生成 JSON，不写入 Milvus")
    args = p.parse_args(argv) 

    llm = llm_server(use_api=(args.use_api == "true"))
    stats = ingest_source_directory(
        args.source_dir,
        llm,
        pdf_strategy=args.pdf_strategy,
        processed_list_path=args.processed_list,
        skip_milvus=args.skip_milvus,
    )

    print("--- ingest_source_to_database ---")
    for k, v in stats.items():
        if k != "errors":
            print(f"  {k}: {v}")
    if stats["errors"]:
        print("  errors:")
        for e in stats["errors"]:
            print(f"    - {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
