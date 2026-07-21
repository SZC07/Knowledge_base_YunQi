# processor/query_processor/nodes/node_item_name_confirm.py

import json
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE, \
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from utils.mongo_history_utils import get_recent_messages, save_chat_message


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        logger.info(f"【{self.name}】节点逻辑")

        # 1 参数校验
        session_id,original_query = self._step_1_validate_param(state) # 会话id和本次问题

        # 2 获取历史会话（记忆）
        history = get_recent_messages(session_id)
        state["history"] = history

        # 3 保存用户信息
        message_id = save_chat_message(session_id,"user",original_query)

        # 4 模型提取主体 rewritten_query(改写问题),item_names(模型识别到的设备主体)
        extract_res = self._step_4_extract_info(original_query, history)
        item_names = extract_res["item_names"] # 可能的设备的主体名称
        state["item_names"] = extract_res["item_names"]
        rewritten_query = extract_res["rewritten_query"] # 模型后改写的问题
        state["rewritten_query"] = rewritten_query
        # 5,6 向量搜索（搜索知识库），搜索结果对齐（整理）
        align_result = {}
        if len(item_names) > 0:
            query_results = self._step_5_vectorize_and_query(item_names)

            # 测试
            # print(f"milvus根据提取到的item_names的匹配结果：")
            # for query_result in query_results:
            #     print(query_result)

            align_result = self._step_6_align_item_names(query_results)
        else:
            logger.info("Node: 未提取到商品名，跳过向量检索")

        # 7 状态state信息整理（确认状态），将第六步的结果对齐到state中
        state = self._step_7_check_confirmation(state, align_result, history)

        # 8 写入历史会话
        self._step_8_write_history(state, session_id, rewritten_query, message_id)
        return state

    # 步骤1
    def _step_1_validate_param(self, state):
        print("step_1: 参数校验")
        session_id = state["session_id"]
        if not session_id:
            raise ValueError("核心参数session_id缺失")
        original_query = state["original_query"]
        if not original_query:
            raise ValueError("核心参数session_id缺失")
        return  session_id,original_query

    def _step_4_extract_info(self, original_query, history):
        print("step_4: 模型提取意图主体")

        # llm
        ai_client = get_llm_client()

        # 拼接上下文（history+original_query）
        history_text = ""
        for msg in history:
            role = msg.get("role")
            content = msg.get("text")
            history_text += f"{role}: {content}\n"

        user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_text,
            query=original_query,
        )
        messages = [
            SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        # 调用llm大模型
        response = ai_client.invoke(messages)
        response_content = response.content
        if response_content.startswith("```json"):
            response_content = response_content.replace("```json", "").replace("```", "")

        # 结果解析
        result = json.loads(response_content)
        print(result)
        if "item_names" not in result:
            result["item_names"] = []
        if "rewritten_query" not in result:
            result["rewritten_query"] = original_query
        if result["item_names"]:
            result["item_names"] = [name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "") for name
                                    in result["item_names"]]

        return result

    def _step_5_vectorize_and_query(self, item_names):
        print("step_5: 向量化并检索,检索出来对应item_name的向量库中的相似的商品名称的列表")
        results : List[Dict] = [] # 大模型识别的可能的名称：milvus的匹配结果集合

        # milvus的客户端，集合名称
        milvus_client = get_milvus_client()
        collection_name = milvus_config.item_name_collection

        # 条件向量化
        embeddings = generate_embeddings(item_names)

        # 相似性匹配
        for i in range(len(item_names)):
            dense_vector = embeddings.get("dense")[i]
            sparse_vector = embeddings.get("sparse")[i]
            reqs = create_hybrid_search_requests(dense_vector=dense_vector, sparse_vector=sparse_vector,limit=5)
            search_res = hybrid_search(
                client=milvus_client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights= (0.8,0.2),
                norm_score=True,
                output_fields=["item_name"]
            )
        # 结果处理
        matches = []
        if search_res and len(search_res) > 0:
            print("此处是意图识别的向量搜索结果")
            print(search_res)
            for hit in search_res[0]:
                # 将search_res中的结果封装到matches中
                matches.append({
                    "item_name": hit.entity.get("item_name"),
                    "score": hit.get("score")
                })
        results.append({
            "extracted_name": item_names[i],
            "matches": matches
        })
        # 返回
        return results

    def _step_6_align_item_names(self, query_results):
        print("step_6: 对齐结果,高分结果和低分结果整合")

        return {
            "confirmed_item_names" : [], # 确认后的高分商品名称（>0.8)
            "options":[] # 可能低分商品名称（>0.6）
        }

    def _step_7_check_confirmation(self, state, align_result, history):
        print("step_7: 状态state信息,根据第六步高分低分对齐结果整理")

        state["answer"] = "" # 如果有高于0.6的可选项options(反问用户，你想问的是xxx设备吗)或者低于0.6（抱歉，未找到相关产品）
        state["item_names"] = [] # 如果有高于0.8的，才封装state进入后续节点
        
        return state

    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        print("step_8: 写入历史会话，更新")


if __name__ == "__main__":

    # 初始化图状态
    init_state = {
        "original_query": "B530这个东西咋整了",
        "session_id":"123"
    }

    # 创建节点对象
    node_item_name_confirm = NodeItemNameConfirm()
    # 执行节点的单元测试
    result = node_item_name_confirm(init_state)
    # 将返回的图状态进行json序列化
    # json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    logger.info(result)