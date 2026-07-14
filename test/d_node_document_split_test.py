import logging

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行...")

        # 1 参数处理
        content,file_title = self.step_1(state)
        # 2 标题切
        sections, title_count, lines_count = self.step_2(content,file_title)
        # 3 无标题（兜底）
        sections = self.step_3(content, sections, title_count, file_title)
        # 4 精切 长切短合
        sections = self._step_4(sections)
        # 5 日志
        self._step_5(lines_count, sections)
        # 6 备份
        self._step_6(state, sections)

        return state

    def step_1(self, state):
        print(1)
        content = state.get("md_content")
        if not content:
            raise StateFieldError(field_name="md_content",expected_type=str)
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title",expected_type=str)

        content = content.replace("\r\n","\n").replace("\r","\n")
        return content,file_title

    def step_2(self, content, file_title):
        print(2)
        # 参数
        sections = [],
        title_count = 0,
        lines = content.split("\n")



        return  sections, title_count, len(lines)

    def step_3(self, content, sections, title_count, file_title):
        print(3)
        if title_count == 0:
            return [{"title":"无标题","file_title":file_title,"content":content}]
        return sections

    def _step_4(self, sections):
        print(4)

        fined_sections = []

        final_sections = self.merge(fined_sections)
        return final_sections

    def _step_5(self, lines_count, sections):
        print(5)

        pass


    def _step_6(self, state, sections):
        print(6)
        pass

    def merge(self, fined_sections):

        return fined_sections



if __name__ == "__main__":
    node = NodeDocumentSplit()
    with open(r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\output\B530\B530_new.md","r",encoding="utf-8") as f:
        md_content = f.read()
    init_state = {
        "md_path": r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\output\B530\B530_new.md",
        "md_content": md_content,
        "file_title": "B530_new",
    }
    print(node.process(init_state))