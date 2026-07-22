# processor/query_processor/nodes/node_search_embedding.py
import json


from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search


class NodeSearchEmbedding(NodeBase):
     """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """

     # 覆盖基类的 name 属性，标识节点名称
     name: str = "node_search_embedding"

     def process(self, state: QueryGraphState) -> QueryGraphState:
         """
         节点逻辑
         :param state: 工作流状态对象
         :return: 更新后的状态对象
         """

         # TODO
         logger.info(f"【{self.name}】节点逻辑")
         # 参数
         item_names = state.get("item_names") # 元数据向量条件
         query = state.get("rewritten_query") # 语义搜索条件

         # 向量化
         embeddings = generate_embeddings([query])
         dense_vector = embeddings.get("dense")[0]
         sparse_vector = embeddings.get("sparse")[0]

         # 获取milvus集合
         milvus_client = get_milvus_client()
         chunks_collection = milvus_config.chunks_collection

         # 请求对象
         reqs = create_hybrid_search_requests(
             dense_vector=dense_vector,
             sparse_vector=sparse_vector,
             expr=f'item_name in {item_names}'
         )

         # 发送请求
         response = hybrid_search(
             client=milvus_client,
             collection_name=chunks_collection,
             reqs=reqs,
             ranker_weights=(0.8,0.2),
             output_fields=["chunk_id", "content", "item_name"]
         )

         print(response[0])


         # return state
         return {"embedding_chunks": response[0] if response else []}  # [[结果解析]]

if __name__ == "__main__":
    node_search_embedding = NodeSearchEmbedding()
    init_state = {
        "item_names": ["兄弟HAK180烫金机", "百度一下"],
        "rewritten_query": "兄弟请帮我查一下HAK180烫金机是什么?"
    }
    process = node_search_embedding.process(init_state)
    print(json.dumps(process,ensure_ascii=False, indent=4))