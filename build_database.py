#!/usr/bin/env python3
"""文档入库工具：解析 source_file/ 目录下的文档并构建向量数据库"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_config
from ingest_source_to_database import DEFAULT_SOURCE_DIR, ingest_source_directory
from llm.llm_server import llm_server


def main():
    print("=" * 60)
    print("文档入库")
    print("=" * 60)

    config = load_config()
    source_dir = str(config.paths.get_source_file_dir())

    print(f"源文件目录: {source_dir}")
    print(f"初始化 LLM...")

    llm = llm_server(use_api=True)

    print("开始处理...")
    print("-" * 60)

    stats = ingest_source_directory(
        source_dir=source_dir,
        llm=llm,
        pdf_strategy="fast",
        skip_milvus=False,
    )

    print("\n处理结果:")
    print(f"  总计: {stats['total_sources']}")
    print(f"  已跳过: {stats['skipped_already_processed']}")
    print(f"  待处理: {stats['pending']}")
    print(f"  成功处理: {stats['chunked_ok']}")
    print(f"  入库条数: {stats['milvus_rows_inserted']}")

    if stats["errors"]:
        print(f"\n错误 ({len(stats['errors'])}):")
        for err in stats["errors"]:
            print(f"  - {err}")

    print("-" * 60)
    print("完成")


if __name__ == "__main__":
    main()
