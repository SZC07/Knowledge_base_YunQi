"""
节点 E：商品名称识别

功能概述：
    1. 从知识库文档中，利用大模型（LLM）自动识别出文档对应的"商品名称"
    2. 将识别出的商品名称回填到每个文档切片中，用于后续检索
    3. 对商品名称进行向量化（稠密向量 + 稀疏向量），为存入 Milvus 向量库做准备

在整个导入流程中的位置：
    知识库文件导入 → 文档解析 → 文档切片 → 【当前节点：商品名称识别】 → 向量化入库

学习要点：
    - LangGraph 状态图节点开发模式：继承 BaseNode，实现 process(state) → 返回 state
    - LLM 调用：通过 ChatOpenAI 调用大模型，构造 SystemMessage + HumanMessage 进行对话
    - 向量化：一个文本可以同时生成稠密向量（语义表示）和稀疏向量（关键词匹配），两者互补
"""

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
    """
    主体识别节点：主体识别与标签提取

    继承自 BaseNode（所有节点的统一基类），必须实现 process(state) 方法。
    LangGraph 框架会通过 __call__ 自动调用 process，并包裹统一的日志和异常处理。

    核心流程（6 步）：
        参数提取 → 上下文拼接 → LLM 识别 → 结果回填 → 向量化 → 存入向量库
    """

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
        file_title, chunks = self.step_1_get_inputs(state)
        # 2 上下文拼接
        context = self.step_2_build_context(file_title, chunks)
        # 3 调用大模型
        item_name = self.step_3_call_llm(file_title, context)
        # 4 回填数据
        self.step_4_update_chunks(state, chunks, item_name)
        # 5 主体名称向量化（返回稠密向量和稀疏向量）
        dense_vector,sparse_vector = self._step_5_generate_vectors(item_name)
        # 6 存入向量数据库
        self.step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)


        return state

    def step_1_get_inputs(self, state):
        """
        步骤 1：从状态中提取并校验必要参数。

        返回值：
            file_title: 文件名（如 "B530产品手册"），是识别商品名称的重要线索
            chunks:     文档切片列表，每个切片是一个 dict，包含 title（章节标题）和 context（正文内容）

        异常处理：
            如果必填字段缺失，抛出 StateFieldError，由基类的 __call__ 方法统一捕获处理。
        """
        print("步骤1：参数处理")
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)
        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)
        return file_title,chunks

    def step_2_build_context(self, file_title, chunks):
        """
        步骤 2：将多个切片拼接成 LLM 的上下文。

        拼接策略：
            1. 只取前 k 个切片（由配置 item_name_chunk_k 控制，避免上下文过长）
            2. 每个切片格式化为：【切片N】\n标题{title}\n内容{context}
            3. 累计字符数超过 chunk_size 时提前截断，确保不超出 LLM 的 token 限制

        这种"前缀截断 + 格式化拼接"是 RAG 场景中构造 LLM 上下文的常见做法。
        """
        print("步骤2：上下文拼接")

        final_context = ""
        # 上线文限制的片数
        k = self.config.item_name_chunk_k
        # 上线文限制的切片长度
        chunk_size = self.config.item_name_chunk_size

        parts:List[Dict] = []
        total_chars = 0
        for index, chunk in enumerate(chunks[:k],start=1):
            chunk_title = chunk.get("title","").strip()
            chunk_context = chunk.get("context","").strip()

            # 格式化
            piece = f"【切片{index}】\n标题{chunk_title}\n内容{chunk_context}"
            parts.append(piece)

            # 计算长度
            total_chars += len(chunk_title)

            # 检测长度
            if total_chars > chunk_size:
                break

        # 截断处理
        context = "\n\n".join(parts).strip()
        final_context = context[:chunk_size]

        return final_context

    def step_3_call_llm(self, file_title, context):
        """
        步骤 3：调用大模型（LLM）识别商品名称。

        核心逻辑：
            1. 如果上下文为空，直接返回文件名作为兜底
            2. 构造提示词（Prompt），告诉 LLM 它的角色和任务
            3. 将 SystemMessage（角色设定）和 HumanMessage（具体任务）组合发送给 LLM
            4. 解析 LLM 返回结果，清洗空格和换行符
            5. 如果 LLM 返回空字符串，回退使用文件名

        学习要点：
            - SystemMessage：定义 AI 的角色和行为规范（"你是一个专业的商品名称识别模型"）
            - HumanMessage：包含具体的输入数据和要求（文件名 + 切片内容 + 输出格式要求）
            - temperature=0 通常用于需要稳定、确定性输出的任务（如信息提取）
            - enable_thinking=False 关闭模型的思考过程，直接获取结果，节省 token
        """
        print("步骤3：调用大模型")
        item_name = ""
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
        # 解析
        item_name = getattr(response, "content").strip() # 主体名称
        item_name = item_name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
        # 兜底
        if not item_name:
            item_name = file_title

        return item_name

    def step_4_update_chunks(self, state, chunks, item_name):
        """
        步骤 4：将识别出的商品名称回填到 state 和所有切片中。

        为什么要回填到每个切片？
            后续检索时，用户搜索某个商品名称，需要匹配到该商品的所有文档切片。
            每个切片都带上 item_name，就能在 Milvus 中建立"商品名 → 切片"的映射关系。
        """
        print("步骤4：回填数据")
        state["item_name"] = item_name
        for chunk in chunks:
            chunk["item_name"] = item_name
        return state

    def _step_5_generate_vectors(self, item_name):
        """
        步骤 5（内部方法）：将商品名称文本转为向量。

        返回两种向量，用于混合检索：
            - dense_vector  （稠密向量）：捕捉语义信息，适合语义搜索（如 "笔记本电脑" 匹配 "笔记本"）
            - sparse_vector （稀疏向量）：捕捉关键词信息，适合精确匹配（如 BM25 算法）

        方法名以 _ 开头，表示这是类的内部方法，不建议外部直接调用。
        """
        print("步骤5：向量化：返回稠密向量和稀疏向量")
        embeddings = generate_embeddings([item_name]) # 稠密向量和稀疏向量
        dense = embeddings["dense"][0]
        sparse = embeddings["sparse"][0]
        return dense,sparse

    def step_6_save_to_milvus(self, state, file_title, item_name, dense_vector, sparse_vector):
        """
        步骤 6：将商品名称及其向量存入 Milvus 向量数据库（预留接口，待实现）。

        Milvus 是一个开源的向量数据库，支持高效的向量相似度搜索。
        存入后，用户可以通过自然语言搜索商品名称，系统返回最相似的结果。
        """
        print("步骤6：存入向量库")
        pass

# ==================== 本地调试入口 ====================
# 直接运行此文件可快速测试节点功能，无需启动完整导入流程。
# 使用方法：
#     1. 准备一个 chunks JSON 文件（前面节点的输出）
#     2. 修改 path 变量指向你的测试文件
#     3. 运行：python e_node_item_name_recognition.py
if __name__ == "__main__":
    node = NodeItemNameRecognition()

    path = r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\output\B530\B530_new_chunks.json"

    with open(path, "r", encoding="utf-8") as f:
        chunks_json_data = f.read()

    init_state={
        "file_title":"B530",
        "chunks":json.loads(chunks_json_data)
    }

    process = node.process(init_state)
    for chunk in process["chunks"]:
        print(f"{chunk['item_name']}章节：{chunk['title']}")

    print(process)




