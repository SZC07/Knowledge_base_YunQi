from dotenv import load_dotenv
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from processor.query_processor.nodes.node_answer_output import NodeAnswerOutput
from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.node_rerank import NodeRerank
from processor.query_processor.nodes.node_rrf import NodeRrf
from processor.query_processor.nodes.node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from processor.query_processor.nodes.node_web_search_mcp import NodeWebSearchMcp
from processor.query_processor.state import QueryGraphState

load_dotenv()


class KBQueryWorkflow:

    # 构造
    def __init__(self):
        print("初始化查询工作流...")
        # 1 工作流状态
        self.workflow = StateGraph(QueryGraphState)

        # 2 实例化所有节点
        self._init_nodes()

        # 3 注册所有节点(添加节点)
        self._register_nodes()

        # 4 设置路由规则(设置边)
        self._setup_routes()

        # 5 编译工作流（懒加载，首次执行时编译）
        self._compiled_app = None

    # 实例化
    def _init_nodes(self):
        print("实例化所有节点...")
        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer_output = NodeAnswerOutput()

    # 注册
    def _register_nodes(self):
        print("注册所有节点...")
        self.workflow.add_node("node_item_name_confirm", self.node_item_name_confirm)
        self.workflow.add_node("node_search_embedding", self.node_search_embedding)
        self.workflow.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde)
        self.workflow.add_node("node_web_search_mcp", self.node_web_search_mcp)
        self.workflow.add_node("node_rrf", self.node_rrf)
        self.workflow.add_node("node_rerank", self.node_rerank)
        self.workflow.add_node("node_answer_output", self.node_answer_output)

    # 路由
    def _setup_routes(self):
        print("设置路由规则...")
        # 入口
        self.workflow.set_entry_point("node_item_name_confirm")
        # self.workflow.add_node(START,"node_item_name_confirm")

        # 条件边
        self.workflow.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm,
            {
                "node_search_embedding": "node_search_embedding",
                "node_search_embedding_hyde": "node_search_embedding_hyde",
                "node_web_search_mcp": "node_web_search_mcp",
                "node_answer_output": "node_answer_output"  # 问题确认后，不进入任何检索分支，直接返回结果
            }
        )

        # 注册边
        self.workflow.add_edge("node_search_embedding", "node_rrf")
        self.workflow.add_edge("node_search_embedding_hyde", "node_rrf")
        self.workflow.add_edge("node_web_search_mcp", "node_rrf")

        self.workflow.add_edge("node_rrf", "node_rerank")
        self.workflow.add_edge("node_rerank", "node_answer_output")
        self.workflow.add_edge("node_answer_output", END)

    # 路由-条件
    def _route_after_item_name_confirm(self, state: QueryGraphState) -> str:
        print("路由-条件...")
        if state.get("answer"):
            return "node_answer_output"
        return ["node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"]

    # 编译图
    def compile(self):
        print("编译图...")
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    # 调用
    def run(self, initial_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        print("调用图...")
        """
        统一执行入口，支持切换invoke/stream
        """
        if not self._compiled_app:
            self.compile()
        if stream:
            return self._compiled_app.stream(initial_state)
        else:
            return self._compiled_app.invoke(initial_state)
if __name__ == "__main__":
    # 调用图
    workflow = KBQueryWorkflow()
    response = workflow.run({"original_query": "华为擎云B530咋鼓捣咋和H3CLA2608室内无线网关弄上啊？","session_id":"123"},stream=True)
    for res in response:
        print(res)

    # 画图
    # print(workflow.compile().get_graph().draw_ascii())
