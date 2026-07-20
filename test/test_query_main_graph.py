from langgraph.graph import StateGraph

from processor.query_processor.state import QueryGraphState


class KBQueryWorkflow:
    def __init__(self):
        self.workflow = StateGraph(QueryGraphState)
        pass