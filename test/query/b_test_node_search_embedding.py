from pymilvus.milvus_client import milvus_client

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client


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

        # 参数处理
        item_names = state.get("item_names") # 过滤条件
        query = state.get("rewritten_query") # 语义搜索

        # 向量化
        embeddings = generate_embeddings([query])
        dense_vec = embeddings["dense"][0]
        sparse_vec = embeddings["sparse"][0]

        # milvus
        milvus_client = get_milvus_client()
        chunks_collection = milvus_config.chunks_collection

        


        return {"embedding_chunks":[] }  # [[结果解析]]