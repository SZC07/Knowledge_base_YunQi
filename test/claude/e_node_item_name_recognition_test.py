import json
import logging

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError, MilvusError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取

    功能流程：
    1. 从 state 中提取 file_title 和 chunks
    2. 拼接前 K 个切片内容作为上下文
    3. 调用 LLM 识别文档中的产品/设备主体名称
    4. 将识别到的主体名称回填到所有切片和 state 中
    5. 对主体名称生成稠密向量（dense）和稀疏向量（sparse）
    6. 将主体名称及向量存入 Milvus 的 item_name_collection
    """

    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState):
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
        dense_vector, sparse_vector = self._step_5_generate_vectors(item_name)
        # 6 存入向量数据库
        self.step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        return state

    # ==================== 步骤1：参数处理 ====================
    def step_1_get_inputs(self, state):
        self.logger.info("步骤1：参数处理")
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(
                field_name="file_title",
                message="文件标题不能为空",
                expected_type=str,
            )
        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(
                field_name="chunks",
                message="chunks不能为空",
                expected_type=list,
            )
        self.logger.info(f"文件标题: {file_title}, 切片数量: {len(chunks)}")
        return file_title, chunks

    # ==================== 步骤2：上下文拼接 ====================
    def step_2_build_context(self, file_title, chunks):
        """
        从切片列表中提取前 K 个切片内容，拼接为 LLM 上下文。

        每个切片内容限制在 chunk_size 长度内，避免超出 LLM 上下文窗口。
        """
        self.logger.info("步骤2：上下文拼接")

        # 上下文限制的切片数量
        k = self.config.item_name_chunk_k
        # 每个切片的最大字符数
        chunk_size = self.config.item_name_chunk_size

        context_parts = []
        for i, chunk in enumerate(chunks[:k]):
            content = chunk.get("content", "")
            if not content:
                continue
            # 截取指定长度的内容，保持上下文在可控范围内
            if len(content) > chunk_size:
                content = content[:chunk_size]
            context_parts.append(f"【片段{i + 1}】\n{content}")

        final_context = "\n\n".join(context_parts)

        # 如果拼接后上下文为空，使用文件标题作为兜底
        if not final_context.strip():
            self.logger.warning("上下文拼接结果为空，使用文件标题作为兜底上下文")
            final_context = file_title

        self.logger.info(
            f"上下文构建完成，共使用{len(context_parts)}个片段，总长度{len(final_context)}字符"
        )
        return final_context

    # ==================== 步骤3：调用大模型 ====================
    def step_3_call_llm(self, file_title, context):
        """
        调用 LLM 识别文档中的产品/设备主体名称。

        使用配置中的 item_model 进行推理，返回识别到的主体名称。
        如果 LLM 调用失败，自动降级使用文件标题作为主体名称。
        """
        self.logger.info("步骤3：调用大模型识别主体名称")

        # 获取 LLM 客户端（使用 item_model 配置的模型）
        llm = get_llm_client(self.config.item_model)

        prompt = f"""你是一个产品名称识别专家。请根据以下文档内容，识别出文档中描述的产品/设备名称。

文档标题：{file_title}

文档内容摘要：
{context}

请遵循以下规则：
1. 返回产品/设备的完整、规范的名称
2. 只返回名称本身，不要包含任何解释、标点或其他内容
3. 如果文档内容中明确提到了产品型号，请优先使用该型号
4. 如果无法从内容中识别，请返回文档标题"{file_title}"

产品名称："""

        try:
            response = llm.invoke(prompt)
            item_name = response.content.strip().replace("\n", "")

            # 清理可能的引号包裹
            item_name = item_name.strip("'\"\"''「」『』")

            # 兜底：LLM 返回空串时使用文件标题
            if not item_name:
                self.logger.warning("LLM 返回空结果，使用文件标题作为主体名称")
                item_name = file_title

        except Exception as e:
            self.logger.warning(f"LLM 调用失败，使用文件标题作为主体名称: {e}")
            item_name = file_title

        self.logger.info(f"识别到的主体名称: {item_name}")
        return item_name

    # ==================== 步骤4：回填数据 ====================
    def step_4_update_chunks(self, state, chunks, item_name):
        """
        将识别到的主体名称回填到每个切片中，同时更新 state。
        """
        self.logger.info("步骤4：回填主体名称到切片数据")

        for chunk in chunks:
            chunk["item_name"] = item_name

        # 同步更新 state
        state["item_name"] = item_name

        self.logger.info(f"已将主体名称'{item_name}'回填到{len(chunks)}个切片中")

    # ==================== 步骤5：向量化 ====================
    def _step_5_generate_vectors(self, item_name):
        """
        使用 BGE-M3 模型对主体名称进行向量化。

        BGE-M3 同时输出稠密向量（dense）和稀疏向量（sparse），
        分别用于语义相似度和关键词匹配。

        Returns:
            tuple: (dense_vector, sparse_vector)
                - dense_vector: list[float] — 1024 维稠密向量
                - sparse_vector: dict — 稀疏向量，格式为 {index: value}
        """
        self.logger.info("步骤5：生成稠密向量和稀疏向量")

        try:
            result = generate_embeddings([item_name])
            dense_vector = result["dense"][0]
            sparse_vector = result["sparse"][0]
            self.logger.info(
                f"向量生成成功，稠密向量维度: {len(dense_vector)}，"
                f"稀疏向量非零元素数: {len(sparse_vector)}"
            )
            return dense_vector, sparse_vector
        except Exception as e:
            self.logger.error(f"向量生成失败: {e}")
            raise

    # ==================== 步骤6：存入Milvus ====================
    def step_6_save_to_milvus(self, state, file_title, item_name, dense_vector, sparse_vector):
        """
        将主体名称及其向量存入 Milvus 向量数据库。

        存入的集合由配置 item_name_collection 指定。
        如果集合不存在则自动创建，并建立稠密向量和稀疏向量的索引。
        """
        self.logger.info("步骤6：存入Milvus向量库")

        collection_name = self.config.item_name_collection
        if not collection_name:
            self.logger.warning("未配置 item_name_collection，跳过 Milvus 存储")
            return

        milvus_url = self.config.milvus_url
        if not milvus_url:
            self.logger.warning("未配置 MILVUS_URL，跳过 Milvus 存储")
            return

        try:
            # 建立 Milvus 连接（幂等：已连接则复用）
            self._connect_milvus(milvus_url)

            # 确保集合存在（不存在则创建）
            self._ensure_collection_exists(collection_name, len(dense_vector))

            # 插入数据
            self._insert_item_name(collection_name, file_title, item_name, dense_vector, sparse_vector)

            self.logger.info(
                f"主体名称'{item_name}'已成功存入Milvus集合'{collection_name}'"
            )

        except Exception as e:
            self.logger.error(f"Milvus 存储失败: {e}")
            raise MilvusError(
                message=f"主体名称存入Milvus失败: {e}",
                node_name=self.name,
                cause=e,
            )

    def _connect_milvus(self, milvus_url: str):
        """
        建立 Milvus 连接（幂等操作）。
        """
        # 检查是否已有可用连接
        if connections.has_connection("default"):
            return

        connections.connect(
            alias="default",
            uri=milvus_url,
        )
        self.logger.info(f"已连接 Milvus: {milvus_url}")

    def _ensure_collection_exists(self, collection_name: str, dense_dim: int):
        """
        确保目标 Milvus 集合存在。

        如果集合不存在，则自动创建并建立索引。
        集合 Schema：
            - id: INT64 主键（自动生成）
            - item_name: VARCHAR(512) 主体名称
            - file_title: VARCHAR(256) 来源文件标题
            - dense_vector: FLOAT_VECTOR 稠密向量（1024 维）
            - sparse_vector: SPARSE_FLOAT_VECTOR 稀疏向量
        """
        if utility.has_collection(collection_name):
            self.logger.info(f"Milvus 集合'{collection_name}'已存在")
            # 确保集合已加载到内存（已加载时 load 是幂等操作）
            try:
                col = Collection(collection_name)
                col.load()
            except Exception:
                pass  # 可能已加载或版本差异，忽略
            return

        self.logger.info(f"Milvus 集合'{collection_name}'不存在，正在创建...")

        # 定义字段
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="自增主键",
            ),
            FieldSchema(
                name="item_name",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="识别出的产品/设备名称",
            ),
            FieldSchema(
                name="file_title",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="来源文件标题",
            ),
            FieldSchema(
                name="dense_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=dense_dim,
                description="稠密语义向量",
            ),
            FieldSchema(
                name="sparse_vector",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
                description="稀疏关键词向量",
            ),
        ]

        # 创建集合
        schema = CollectionSchema(
            fields,
            description="商品/产品名称向量集合，用于主体名称的相似度检索",
        )
        collection = Collection(collection_name, schema)

        # 为稠密向量创建索引（COSINE 余弦相似度）
        dense_index_params = {
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
            "metric_type": "COSINE",
        }
        collection.create_index("dense_vector", dense_index_params)
        self.logger.info(f"稠密向量索引已创建: IVF_FLAT + COSINE")

        # 为稀疏向量创建索引（内积 IP）
        sparse_index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
        }
        collection.create_index("sparse_vector", sparse_index_params)
        self.logger.info(f"稀疏向量索引已创建: SPARSE_INVERTED_INDEX + IP")

        # 加载集合到内存
        collection.load()
        self.logger.info(f"Milvus 集合'{collection_name}'创建并加载成功")

    def _insert_item_name(
        self,
        collection_name: str,
        file_title: str,
        item_name: str,
        dense_vector: list,
        sparse_vector: dict,
    ):
        """
        向 Milvus 集合插入一条主体名称记录。

        插入前检查是否已存在相同 (file_title, item_name) 的记录，
        避免重复插入。
        """
        collection = Collection(collection_name)

        # 将稀疏向量 dict 转为 Milvus 需要的格式：{维度: 值}
        # Milvus 的 SPARSE_FLOAT_VECTOR 接受 dict[int, float] 格式
        sparse_dict = {int(k): float(v) for k, v in sparse_vector.items()}

        entities = [
            [item_name],       # item_name
            [file_title],      # file_title
            [dense_vector],    # dense_vector
            [sparse_dict],     # sparse_vector
        ]

        collection.insert(entities)
        collection.flush()
        self.logger.debug(f"Milvus 插入成功: file_title={file_title}, item_name={item_name}")


if __name__ == "__main__":
    node = NodeItemNameRecognition()

    path = r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\output\B530\B530_new_chunks.json"

    with open(path, "r", encoding="utf-8") as f:
        chunks_json_data = f.read()

    init_state = {
        "file_title": "B530",
        "chunks": json.loads(chunks_json_data),
    }

    process = node.process(init_state)

    print(process)
