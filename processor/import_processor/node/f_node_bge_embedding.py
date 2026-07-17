import json
from typing import List, Dict


from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState):
        # 1 参数校验
        chunks = self._step_1_validate_paths(state)

        # 2 数据向量化
        output_data = self._step_generate_embeddings(chunks)
        for item in output_data:
            item_name = item.get("item_name")
            content = item.get("content")
            print(f"{item_name}")
            sparse_vector = item.get("sparse_vector")
            print(sparse_vector)
        path = r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\output\B530\B530_new_new_new_chunks.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                output_data,
                f,
                ensure_ascii=False,
                indent=2
            )

        # 3 返回结果
        state["chunks"] = output_data

        return state

    # 步骤1
    def _step_1_validate_paths(self, state: ImportGraphState) -> List[Dict]:
        print("node_bge_embedding: 步骤1：参数校验")
        # 校验参数
        chunks = state.get("chunks")

        if not chunks:
            raise ValueError("参数错误：chunks")

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        return chunks

    # 步骤2
    def _step_generate_embeddings(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        将item_name和content转化为向量数据(稀疏和稠密)
        """
        print("node_bge_embedding: 步骤2：数据向量化")
        output_data = []
        batch_size = 5  # 批量处理
        for i in range(0, len(chunks), batch_size):
            five_ready_xlh_texts = []
            five_texts = chunks[i:i + batch_size]  # 第一次从0块取到4块，一共取5块
            for doc in five_texts:
                item_name = doc["item_name"]
                content = doc["content"]
                five_ready_xlh_texts.append(f"{item_name}\n{content}" if item_name else content)

            embeddings = generate_embeddings(five_ready_xlh_texts)  # 向量化结果

            for index, doc in enumerate(five_texts):
                item = doc.copy()
                dense = embeddings["dense"][index]
                item["dense_vector"] = dense
                sparse = embeddings["sparse"][index]
                item["sparse_vector"] = sparse
                output_data.append(item)

        return output_data


if __name__ == "__main__":
    node = NodeBGEEmbedding()
    with open(r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\output\B530\B530_new_new_chunks.json", "r", encoding="utf-8") as f:
        chunks_content = f.read()

    json_state = json.loads(chunks_content)

    init_state = {
        "chunks": json_state
    }

    response = node(init_state)
    print(response)
    dumps = json.dumps(response, ensure_ascii=False, indent=4)
    print(dumps)
