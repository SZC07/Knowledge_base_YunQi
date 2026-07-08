from langgraph import graph
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

    # 懒加载，只有第一次运行创建graph
    @property
    def graph(self):
        if self.__compiled_graph is None:
            self.__compiled_graph = self.graph_build()
            pass
        return self.__compiled_graph

    @staticmethod
    def router_edge(state: ImportGraphState):
        if state.get("is_md_read_enabled") == True:
            return "c_node_md_img"
        elif state.get("is_pdf_read_enabled") == True:
            return "b_node_pdf_to_md"
        else:
            return END


    def graph_build(self):
        # 构建graph
        graph = StateGraph(ImportGraphState)

        # 构建节点
        graph.add_node("a_node_entry", NodeEntry())
        graph.add_node("b_node_pdf_to_md", NodePDFToMD())
        graph.add_node("c_node_md_img", NodeMDImg())
        graph.add_node("d_node_document_split", NodeDocumentSplit())
        graph.add_node("e_node_item_name_recognition", NodeItemNameRecognition())
        graph.add_node("f_node_bge_embedding", NodeBGEEmbedding())
        graph.add_node("g_node_import_milvus", NodeImportMilvus())

        # 定义起始节点
        graph.set_entry_point("a_node_entry")

        # 构建边
        graph.add_conditional_edges(
            "a_node_entry",
            self.router_edge,
            {
                "b_node_pdf_to_md":"b_node_pdf_to_md",
                "c_node_md_img":"c_node_md_img"
            }
        )

        # 编译
        graph.compile = graph.compile()
        return graph.compile

    def run(self,state: ImportGraphState,stream = False):
        if stream :
            return self.graph.stream(state,stream_mode="values")
        if not stream:
            return self.graph.invoke(state)

if __name__ == "__main__":
    setup_logging()

    workflow = KBImportWorkflow()
    init_state = {
        "import_file_path": r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\doc\H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf"
    }
    for event in workflow.run(init_state,stream=True):
        print(event)
