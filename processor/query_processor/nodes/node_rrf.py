# processor/query_processor/nodes/node_rrf.py
import json
from typing import List, Tuple

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger


class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web）进行加权融合排序。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        # 1 参数处理
        embedding_chunks = state.get("embedding_chunks")
        hyde_embedding_chunks = state.get("hyde_embedding_chunks")

        embedding_chunks = [doc.get("entity") for doc in embedding_chunks]
        hyde_embedding_chunks = [doc.get("entity") for doc in hyde_embedding_chunks]

        # 2 封装要融合的数据
        rrf_inputs = [
            (embedding_chunks, 1.0),
            (hyde_embedding_chunks, 1.0),
        ]

        # 3 使用rrf算法公式对要融合的数据进行融合
        rrm_merge_results = self._rrf_merge(rrf_inputs)

        # 4 返回结果处理
        rrf_chunks = [doc for doc, _ in rrm_merge_results]  # 只要文档不要分
        state["rrf_chunks"] = rrf_chunks
        print(f"rrf_chunks: {rrf_chunks}")
        return state

    def _rrf_merge(self, rrf_inputs:List[Tuple],k :int = 60,max_results: int = 60):
        print("向量搜索和假设性搜索的融合函数")
        chunk_scores = {}
        chunk_data = {}

        # 循环处理融合数据
        for rrf_input,weight in rrf_inputs:
            for rank,doc in enumerate(rrf_input):
                chunk_id = doc.get("chunk_id")
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id,0.0) + weight/(k+rank)
                chunk_data.setdefault(chunk_id,doc)

        # 未排序的结果
        unsorted_results = [(chunk_data[cid],score) for cid,score in chunk_scores.items()] # List[Tuple(doc,score)]

        # 排序结果
        sorted_results = sorted(
            unsorted_results,
            key=lambda x: x[1],
            reverse=True
        )

        # 返回rrf排序结果
        return sorted_results[:max_results] if max_results else sorted_results

if __name__ == "__main__":
    # 模拟两路检索结果
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#3"}},
        ]
    }
    node_rrf = NodeRrf()
    result = node_rrf.process(mock_state)
    logger.info(json.dumps(result, indent=4))