import json
import logging
from typing import List, Dict, Any

from pymilvus import DataType
from sympy.interactive.session import enable_automatic_symbols

from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, MilvusError
from processor.import_processor.state import ImportGraphState
from utils.milvus_utils import get_milvus_client, escape_milvus_string


class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化
    """

    name = "node_import_milvus"

    def process(self, state: ImportGraphState):

        # 1 数据校验
        chunks_json_data, vector_dimension = self._step_1_check_inputs(state)

        # 2 客户端和集合准备
        milvus_client = self._step_2_prepare_collection(vector_dimension)

        # 3 清理可能的冗余数据(幂等性)
        self._step_3_clean_old_data(milvus_client, chunks_json_data)

        # 4 数据入库(字段+索引),返回数据库主键
        update_chunks = self._step_4_insert_data(milvus_client, chunks_json_data)
        # for chunk in update_chunks:
        #     print(f"chunk_id:{chunk['chunk_id']}")
        # 参照前一步骤生成json文档

        # 5 更新状态
        state["chunks"] = update_chunks

        return state

    # 步骤1 数据校验
    def _step_1_check_inputs(self, state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        print("node_import_milvus: 步骤1：数据校验")
        """
        检查
        """

        # 处理文本块
        chunks_json_data = state.get("chunks")
        if not chunks_json_data:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        # 向量维度
        dimenstion = len(chunks_json_data[0]["dense_vector"])
        print(f"向量维度：{dimenstion}")

        return chunks_json_data, dimenstion

    # 步骤2 客户端和集合准备
    def _step_2_prepare_collection(self, vector_dimension: int):
        print("node_import_milvus: 步骤2：结构准备")
        """
        milvus客户端+集合准备
        """
        # 1 客户端
        milvus_client = get_milvus_client()

        # 2 集合
        collection_name = milvus_config.chunks_collection
        if not milvus_client.has_collection(collection_name):
            self.create_chunks_collection(milvus_client, collection_name, vector_dimension)

        return milvus_client

    # 步骤3 清理旧数据
    def _step_3_clean_old_data(self, client, chunks_json_data):
        print("node_import_milvus: 步骤3：清理旧数据")
        # 1. 获取查询条件
        file_title = chunks_json_data[0].get("file_title")

        # 2. 执行幂等清理
        self.clear_chunks_by_file_title(client, file_title)


    # 步骤4 插入数据
    def _step_4_insert_data(self, milvus_client, chunks_json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print("node_import_milvus: 步骤4：插入数据")
        """
        批量插入milvus
        """

        # 数据处理
        data_to_insert=[]
        for item in chunks_json_data:
            item_copy = item.copy()
            if "part" not in item_copy:
                item_copy["part"] = 0
            data_to_insert.append(item_copy)

        # 批量插入
        insert_result = milvus_client.insert(collection_name=milvus_config.chunks_collection, data=data_to_insert)
        insert_count = insert_result.get("insert_count") # 插入数据的条数
        print(f"插入数据条数：{insert_count}")

        # 回写主键
        insert_ids = insert_result.get("ids")
        for index,item in enumerate(chunks_json_data):
            item["chunk_id"] = str(insert_ids[index])

        # 返回带有主键的chunks
        return chunks_json_data

    # 步骤2方法1
    def create_chunks_collection(self, milvus_client, collection_name, vector_dimension):
        print("node_import_milvus: 步骤2：创建集合{collection_name}")
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)

        # 字段声明
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # 切片内容
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=100)  # 切片标题
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=100)  # 父标题
        schema.add_field(field_name="part", datatype=DataType.INT8)  # 分片编号
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=100)  # 源文件标题
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=100)  # 商品名称（幂等性依据）
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)  # 稠密向量

        # 索引声明
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )

        # 创建集合
        milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


    # 步骤3方法1
    def clear_chunks_by_file_title(self, client, file_title):
        print("node_import_milvus: 步骤3：清理旧数据")
        try:
            file_title = escape_milvus_string(file_title)
            client.delete(collection_name=milvus_config.chunks_collection, filter=f"file_title=='{file_title}'")
        except Exception as e:
            self.logger.error(f"Milvus 数据删除失败: {str(e)}")
            raise MilvusError(f"Milvus 数据删除失败: {str(e)}")


if __name__ == "__main__":
    # setup_logging()

    json_path = r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\output\B530\B530_new_new_new_chunks.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)

    init_state = {
        "chunks": state
    }

    # 执行核心处理流程
    node_import_milvus = NodeImportMilvus()
    result = node_import_milvus(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))
