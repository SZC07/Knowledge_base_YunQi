import json
from pathlib import Path
from typing import List, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pymilvus import DataType
from sympy.interactive.session import enable_automatic_symbols

from config.lm_config import lm_config
from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client, escape_milvus_string


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取
    """

    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState):
        # 1 参数处理
        file_title, chunks = self._step_1_get_inputs(state)

        # 2 上下文拼接
        context = self._step_2_build_context(file_title, chunks)

        # 3 模型识别(总结)
        item_name = self._step_3_call_llm(file_title, context)
        print(f"iterm_name:{item_name}")

        # 4 回填数据(item_name - > chunks)
        self._step_4_update_chunks(state, chunks, item_name)
        path = r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\output\B530\B530_new_new_chunks.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                chunks,
                f,
                ensure_ascii=False,
                indent=2
            )

        # 5 主体名称向量化（稠密，稀疏）
        dense_vector, sparse_vector = self._step_5_generate_vectors(item_name)
        print(f"dense_vector:{dense_vector}")
        print(f"sparse_vector:{sparse_vector}")

        # 6 存入milvus向量库
        self._step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        return state

    def _step_1_get_inputs(self, state) -> (str, List[Dict]):
        print("node_item_name_recognition: 步骤1：参数处理")
        file_title = state["file_title"]
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        chunks = state["chunks"]
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        return file_title, chunks

    def _step_2_build_context(self, file_title, chunks: List[Dict]) -> str:
        print("node_item_name_recognition: 步骤2：上下文拼接")
        # 上线文限制的片数
        k = self.config.item_name_chunk_k
        # 上线文限制的切片长度
        chunk_size = self.config.item_name_chunk_size

        parts: List[Dict] = []
        total_chars = 0
        for index, chunk in enumerate(chunks[:k], start=1):
            chunk_title = chunk.get("title", "").strip()
            chunk_content = chunk.get("content", "").strip()

            # 格式化
            piece = f"【切片{index}】\n标题{chunk_title}\n内容：{chunk_content}"
            parts.append(piece)

            # 计算长度
            total_chars += len(piece)

            # 检测长度
            if total_chars > chunk_size:
                break

        # 截断处理
        context = "\n\n".join(parts).strip()
        final_context = context[:chunk_size]
        return final_context

    def _step_3_call_llm(self, file_title, context) -> str:
        print("node_item_name_recognition: 步骤3：模型识别")

        if not context:
            return file_title

        # llm
        llm_ai = ChatOpenAI(
            model=lm_config.llm_model,
            api_key=lm_config.api_key,
            base_url=lm_config.base_url,
            temperature=lm_config.llm_temperature,
            extra_body={"enable_thinking": False}
        )

        # 提示词
        prompt = f"""
                请从以下信息中识别出商品名称与型号：
                文件名：{file_title}
                
                正文切片（用于辅助识别）：
                {context}
                
                要求：
                1. 返回内容为字符串形式，最好是带品牌、型号和名称的完整商品名称。比如：苏伯尓5000W大功率电磁炉；
                2. 返回结果应该只包含商品名称，不要添加任何解释或其他内容；
                3. 如果无法识别商品名称,请返回空字符串。
        """
        message = [
            SystemMessage("你是一个专业的商品名称识别模型，请根据提供的信息，识别商品名称。名称最好不要超过20个字"),
            HumanMessage(content=prompt)
        ]

        # 调用
        response = llm_ai.invoke(message)

        # 解析，数据清洗
        item_name = getattr(response, "content", "").strip()  # 主体名称
        item_name = item_name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")

        # 兜底
        if not item_name:
            item_name = file_title

        return item_name

    def _step_4_update_chunks(self, state, chunks, item_name):
        print("node_item_name_recognition: 步骤4：回填数据")
        state["item_name"] = item_name
        for chunk in chunks:
            chunk["item_name"] = item_name
        return state

    def _step_5_generate_vectors(self, item_name):
        print("node_item_name_recognition: 步骤5：主体名称向量化,返回稠密和稀疏数据")
        embeddings = generate_embeddings([item_name])  # 稠密和稀疏向量
        dense = embeddings["dense"][0]
        sparse = embeddings["sparse"][0]
        return dense, sparse

    def _step_6_save_to_milvus(self, state, file_title, item_name, dense_vector, sparse_vector):
        print("node_item_name_recognition: 步骤6：存入milvus向量库")

        milvus_url = milvus_config.milvus_url  # 链接地址
        collection_name = milvus_config.item_name_collection  # 表名
        # 校验
        if not milvus_url or not collection_name:
            raise Exception("Milvus配置错误")

        # 链接端
        client = get_milvus_client()

        # 字段(数据结构)
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(
            field_name="pk",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=100
        )
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=100
        )
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # 索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",  # 分组+精确搜索
            metric_type="COSINE",
            params={"nlist": 128}  # 聚类数（影响检索精度/速度）
        )
        index_params.add_index(
            field_name="sparse_vector",  # 字段名
            index_name="sparse_vector_index",  # 索引名
            index_type="SPARSE_INVERTED_INDEX",  # 索引类型
            metric_type="IP",  # 相似度计算方式（内积）
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                # 高效的稀疏检索算法

                "normalize": True,
                # ↑ L2 归一化，让内积 (IP) 等价于余弦相似度

                "quantization": "none"
                # ↑ 关闭量化，保持原始精度：模型生成的向量已经压缩的一半的精度了（BGE_FP16=1），这里就不再压缩了
                # "quantization": "none" → 存储原始向量，不压缩
                # "quantization": "sq8" → 存储压缩后的向量（8-bit 量化
            })

        # 建表
        if not client.has_collection(collection_name):
            client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

        # 幂等性清理同名表数据(collection)
        safe_item_name = escape_milvus_string(item_name)
        filter_expr = f'item_name=="{safe_item_name}"'
        client.delete(collection_name=collection_name, filter=filter_expr)

        # 插入一条
        data = {
            "file_title": file_title,
            "item_name": item_name,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector
        }
        client.insert(collection_name, [data])
        client.load_collection(collection_name)  # 将表数据从存储引擎加载到搜索引擎，为了将来查询相似度用

        state["item_name"] = item_name
        return state


if __name__ == '__main__':
    node = NodeItemNameRecognition()

    path = r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\output\B530\B530_new_chunks.json"

    with open(path, "r", encoding="utf-8") as f:
        chunks_json_data = f.read()

    init_state = {
        "file_title": "B530",
        "chunks": json.loads(chunks_json_data)
    }

    process = node.process(init_state)
    for chunk in process["chunks"]:
        print(f'{chunk.get("title")}章节：{chunk.get("item_name")}')

    print(process)
