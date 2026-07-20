import time

from django.core.serializers import json
from langchain_core.runnables import graph
from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.import_processor.base import setup_logging
from processor.import_processor.node.a_node_entry import NodeEntry
from processor.import_processor.node.b_node_pdf_to_md import NodePDFToMD
from processor.import_processor.node.c_node_md_img import NodeMDImg
from processor.import_processor.node.d_node_document_split import NodeDocumentSplit
from processor.import_processor.node.e_node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.node.f_node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.node.g_node_import_milvus import NodeImportMilvus
from processor.import_processor.state import ImportGraphState


class KBImportWorkflow:
    def __init__(self):
        self.__compiled_graph = None

    """懒加载：只在第一次使用时编译图"""
    @property
    def graph(self):
        if self.__compiled_graph is None:
            self.__compiled_graph = self.build_graph()
        return self.__compiled_graph

    @staticmethod
    def router_after_entry(state:ImportGraphState):
        if state.get("is_pdf_read_enabled"):
            return "b_node_pdf_to_md"
        elif state.get("is_md_read_enabled"):
            return "c_node_md_img"
        else:
            return END


    def build_graph(self):
        # 构建graph
        graph = StateGraph(ImportGraphState)

        # 构建节点
        graph.add_node("a_node_entry",NodeEntry())
        graph.add_node("b_node_pdf_to_md",NodePDFToMD())
        graph.add_node("c_node_md_img",NodeMDImg())
        graph.add_node("d_node_document_split",NodeDocumentSplit())
        graph.add_node("e_node_item_name_recognition",NodeItemNameRecognition())
        graph.add_node("f_node_bge_embedding",NodeBGEEmbedding())
        graph.add_node("g_node_import_milvus",NodeImportMilvus())

        # 构建起始节点
        graph.set_entry_point("a_node_entry")

        # 构建边
        graph.add_conditional_edges(
            "a_node_entry",
            self.router_after_entry,
            {
                "b_node_pdf_to_md":"b_node_pdf_to_md",
                "c_node_md_img":"c_node_md_img",
                END:END
            }
        )
        graph.add_edge("b_node_pdf_to_md","c_node_md_img")
        graph.add_edge("c_node_md_img","d_node_document_split")
        graph.add_edge("d_node_document_split","e_node_item_name_recognition")
        graph.add_edge("e_node_item_name_recognition","f_node_bge_embedding")
        graph.add_edge("f_node_bge_embedding","g_node_import_milvus")
        graph.add_edge("g_node_import_milvus",END)

        # 编译
        return graph.compile()

    def run(self, state:ImportGraphState,stream = False):
        if stream:
            return self.graph.stream(state,stream_mode = "values")
        else:
            return self.graph.invoke(state)

if __name__ == "__main__":
    # 启用日志
    setup_logging()

    # 定义初始状态
    init_state = {"import_file_path": r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\doc\PantumP3500用户手册zh_CNV1.2_1644316283788.pdf"}
    workflow = KBImportWorkflow()

    # 方式一：实例化使用
    for event in workflow.run(init_state,stream = True):
        print(event)

    # 方式二：非流式执行
    # final_state = workflow.run(init_state,stream = False)
    # print(json.dumps(final_state,ensure_ascii=False, indent=4))

