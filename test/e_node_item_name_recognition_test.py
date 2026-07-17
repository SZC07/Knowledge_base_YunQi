import json
import logging
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from modelscope.models.nlp.space.model import generator

from config.lm_config import lm_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings


class NodeItemNameRecognition(BaseNode):
    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState):
        """
        节点主流程入口，按顺序执行 6 个子步骤。

        Args:
            state: 导入流程图的状态字典（类似全局上下文），包含 file_title、chunks 等字段。
                   各步骤可以读写 state，实现步骤间的数据传递。

        Returns:
            更新后的 state 字典，LangGraph 会将其传递给下一个节点。
        """
        logging.info(f"{self.name}节点开始执行...")

        # 1 参数处理
        file_name, chunks = self.step_1_inputs(state)
        # 2 上下文拼接
        context = self.step_2_pinjie(file_name, chunks)
        # 3 调用大模型
        item_name = self.step_3_llm(file_name, context)
        # 4 回填
        self.step_4_update(state, chunks, item_name)
        # 5 向量化，返回稠密稀疏
        dense_vector, sparse_vector = self.step_5_xl(item_name)
        # 6 入向量库
        self.step_6_insert(dense_vector, sparse_vector)

        return state

    def step_1_inputs(self, state):
        print(1)
        file_name = state.get("file_name")
        if not file_name:
            raise StateFieldError("file_name")

        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError("chunks")

        return file_name, chunks

    def step_2_pinjie(self, file_name, chunks):
        print(2)
        context = ""
        final_context = ""
        # 上线文限制的片数
        k = lm_
        # 上线文限制的切片长度


            # 格式化

            # 计算长度

            # 检测长度

        # 截断处理

        return context

    def step_3_llm(self, file_name, context):
        print(3)
        item_name = ""

        return item_name

    def step_4_update(self, state, chunks, item_name):
        print(4)
        pass

    def step_5_xl(self, item_name):
        print(5)
        return None, None

    def step_6_insert(self, dense_vector, sparse_vector):
        print(6)
        pass


if __name__ == "__main__":
    node = NodeItemNameRecognition()
    path = r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\output\B530\B530_new_chunks.json"
    with open(path, "r", encoding="utf-8") as f:
        chunks_json_data = f.read()

    i_state = {
        "file_title": "B530",
        "chunks": json.loads(chunks_json_data)
    }

    process = node.process(i_state)
    print(process)
