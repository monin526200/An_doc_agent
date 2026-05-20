from pymilvus import MilvusClient, DataType, Function, FunctionType
from sentence_transformers import SentenceTransformer
import os
import json

chinese_paper_milvus_uri = 'rag/Chinese_paper_db.db'
english_paper_milvus_uri = 'rag/English_paper_db.db'

embedding_model_path_zh = 'llm/model/bge-small-zh-v1.5'
embedding_model_path_en = 'llm/model/bge-small-en-v1.5'

DIM = 384
metric_type = "IP"


def check_doc(docs: list):
    keys = ['doc_name', 'content', 'category']
    default_key = {
        'doc_name': 'Unknown',
        'content': 'None',
        'category': 'others'
    }
    for doc in docs:
        for key in keys:
            if key not in doc:
                doc[key] = default_key[key]
        if isinstance(doc.get('category'), str):
            doc['category'] = [doc['category']]
    return docs


def read_docs(doc_path: str):
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
    except Exception as e:
        raise RuntimeError(f'文件读取错误：{e}')
    return docs


def ensure_paper_collection(client: MilvusClient, *, rebuild: bool = False) -> None:
    """若不存在则创建 paper_data；rebuild=True 时先删除再建。"""
    if client.has_collection('paper_data') and rebuild:
        client.drop_collection('paper_data')
    if client.has_collection('paper_data'):
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name='embedding', datatype=DataType.FLOAT_VECTOR, dim=DIM)

    schema.add_field(field_name='doc_name', datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name='content', datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name='category', datatype=DataType.ARRAY, element_type=DataType.VARCHAR,
                     max_capacity=10,
                     max_length=64)
    schema.add_field(field_name='sparse_vec', datatype=DataType.SPARSE_FLOAT_VECTOR)

    bm25_function = Function(
        function_type=FunctionType.BM25,
        input_field_names='content',
        output_field_names='sparse_vec',
        name='text_bm25_emb'
    )
    schema.add_function(bm25_function)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name='embedding',
        index_type='FLAT',
        metric_type='IP'
    )
    index_params.add_index(
        field_name='sparse_vec',
        index_type='SPARSE_INVERTED_INDEX',
        metric_type='BM25',
        params={'drop_ratio_build': 0.2}
    )

    client.create_collection(
        collection_name='paper_data',
        schema=schema,
        index_params=index_params,
        metric_type=metric_type
    )


def _encode_and_insert(client: MilvusClient, model, docs: list) -> int:
    docs = check_doc(docs)
    content = [d['content'] for d in docs]
    embedding = model.encode(content)
    data = [
        {
            'embedding': emb,
            'content': d['content'],
            'doc_name': d['doc_name'],
            'category': d['category']
        } for d, emb in zip(docs, embedding)
    ]
    client.insert(
        collection_name='paper_data',
        data=data
    )
    return len(data)


def insert_json_into_milvus(uri: str, json_paths: list[str], text_lg: str) -> int:
    """
    向已有 Milvus 库增量插入若干 JSON chunk 文件（每个 JSON 为 list[chunk]）。
    collection 不存在时自动创建。返回插入的向量条数。
    """
    paths = [p for p in json_paths if p and os.path.isfile(p)]
    if not paths:
        return 0

    client = MilvusClient(uri)
    ensure_paper_collection(client, rebuild=False)

    if text_lg == 'zh':
        model = SentenceTransformer(embedding_model_path_zh)
    elif text_lg == 'en':
        model = SentenceTransformer(embedding_model_path_en)
    else:
        client.close()
        raise RuntimeError("语言类型错误，应为 zh 或 en")

    total = 0
    try:
        for path in paths:
            docs = read_docs(path)
            if not isinstance(docs, list):
                raise RuntimeError(f'JSON 格式应为 chunk 列表: {path}')
            total += _encode_and_insert(client, model, docs)
    finally:
        client.close()
    return total


def build_milvus_database(uri: str, doc_dir: str, text_lg: str = 'zh', rebuild=False):
    if not os.path.exists(doc_dir):
        raise FileNotFoundError(f"路径不存在: {doc_dir}")

    doc_path = []
    if os.path.isfile(doc_dir):
        doc_path = [doc_dir]
    elif os.path.isdir(doc_dir):
        doc_path = [
            os.path.join(doc_dir, f)
            for f in os.listdir(doc_dir)
            if f.lower().endswith('.json')
        ]
    if len(doc_path) == 0:
        raise RuntimeError('文件目录下没有json文件')

    client = MilvusClient(uri)
    ensure_paper_collection(client, rebuild=rebuild)

    if text_lg == 'zh':
        model = SentenceTransformer(embedding_model_path_zh)
    elif text_lg == 'en':
        model = SentenceTransformer(embedding_model_path_en)
    else:
        client.close()
        raise RuntimeError("语言类型错误")

    try:
        for path in doc_path:
            docs = read_docs(path)
            _encode_and_insert(client, model, docs)
    finally:
        client.close()
