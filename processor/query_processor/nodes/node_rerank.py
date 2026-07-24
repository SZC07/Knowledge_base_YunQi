# processor/query_processor/nodes/node_rerank.py
from typing import List, Dict, Any

from modelscope.preprocessors.templates.utils import upper_bound

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.reranker_http_utils import rerank_documents


class NodeRerank(NodeBase):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        # 1 将web_doc和rrf_chunks节点的数据合并
        merged_multi_docs: List[Dict[str,Any]] = self._step_1_merge_multi_source_docs(state)
        print(merged_multi_docs)
        # 2 调用排序模型对合并后结果进行合并排序
        reranked_docs: List[Dict[str, Any]] = self._step_2_rerank_merged_docs(state, merged_multi_docs)
        # 3 断崖测试（分数差距，分数比率）去掉过低的匹配结果
        cutoff_docs = self._step_3_cliff_cutoff(reranked_docs)
        # 4 返回结果封装
        print("步骤4：返回结果封装")
        state["reranked_docs"] = cutoff_docs

        return state

    def _step_1_merge_multi_source_docs(self, state):
        print("步骤1：将web_doc和rrf_chunks节点的数据合并")
        final_docs = []

        rrf_chunks = state.get("rrf_chunks")
        web_docs = state.get("web_search_docs")

        for rrf_doc in rrf_chunks:
            format_rrf_doc = {
                "content": rrf_doc.get("content"),
                "title": rrf_doc.get("title"),
                "chunk_id": rrf_doc.get("chunk_id"),
                "url": None,
                "source": "local",
            }
            final_docs.append(format_rrf_doc)

        for web_doc in web_docs:
            format_web_doc = {
                "content": web_doc.get("snippet"),
                "title": web_doc.get("title"),
                "chunk_id": None,
                "url": web_doc.get("url"),
                "source": "web",
            }
            final_docs.append(format_web_doc)
        
        return final_docs

    def _step_2_rerank_merged_docs(self, state, merged_multi_docs):
        print("步骤2：调用排序模型对合并后结果进行排序")

        rewritten_query = state.get("rewritten_query")
        contents = [doc.get("content") for doc in merged_multi_docs]

        # 调用排序模型根据rewritten_query对contents进行排序
        rerank_scores = rerank_documents(rewritten_query, contents)
        scored_docs = [{**doc,"score":score} for doc ,score in zip(merged_multi_docs, rerank_scores) ]

        # scored_docs = []
        # for doc, score in zip(merged_multi_docs, rerank_scores):
        #     scored_docs.append({
        #         "content": doc.get("content"),
        #         "title": doc.get("title"),
        #         "chunk_id": doc.get("chunk_id"),
        #         "url": doc.get("url"),
        #         "source": doc.get("source"),
        #         "score": float(score),
        #     })

        scored_docs = sorted(
            scored_docs,
            key = lambda x :x["score"],
            reverse = True
        )
        print(scored_docs)
        return scored_docs

    def _step_3_cliff_cutoff(self, reranked_docs):
        print("步骤3：断崖测试(分数差距，分数比率)去掉过低的匹配结果")
        # 最多要多少条结果
        upper_bound = min(10,len(reranked_docs))
        # 最多要多少条结果
        lower_bound = max(3,len(reranked_docs))
        # 断崖位置
        cutoff_pos = upper_bound
        for index in range(lower_bound-1,upper_bound-1):
            # 当前分数
            cutoff_score = reranked_docs[index].get("score")
            # 下一个分数
            next_score = reranked_docs[index+1].get("score")
            # 分差
            abs_gap = cutoff_score - next_score
            # 比率(防止分母为0)
            rel_gap = abs_gap/(abs(cutoff_score)+1e-10)

            if abs_gap >=0.5 or rel_gap >=0.25:
                cutoff_pos = index+1 # 找到断崖位置
                break

        return reranked_docs[:cutoff_pos]


if __name__ == "__main__":
    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {
                "chunk_id": "local_1",
                "title": "主板维修手册",
                "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"
            },
            {
                "chunk_id": "local_2",
                "title": "闲聊",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"
            },
        ],
        "web_search_docs": [
            {
                "url": "https://example.com/repair",
                "title": "短路查修指南",
                "snippet": "主板通电前先打各主供电电感对地阻值，阻值偏低就是短路。"
            },
            {
                "url": "https://example.com/news",
                "title": "科技新闻",
                "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"
            },
        ],
    }

    node_rerank = NodeRerank()
    result = node_rerank(mock_state)
    logger.info(result)
    # logger.info(json.dumps(result, indent=4))