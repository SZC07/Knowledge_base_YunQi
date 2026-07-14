import logging
import re
from typing import List, Dict

from langchain_core.messages import content
from langchain_text_splitters import RecursiveCharacterTextSplitter
from modelscope.models.multi_modal.guided_diffusion import script

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.import_config import get_config
from processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行...")

        # 1 参数处理
        content,file_title = self._step_1_get_inputs(state)
        # 2 标题切(初切)
        sections,title_count,lines_count= self._step_2_split_by_title(content, file_title)
        # 3 无标题兜底(默认标题)
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)
        # 4 块精细化处理(长切短合)
        sections = self._step_4_refine_chunks(sections)
        # 测试
        for section in sections:
            # print(f"{section['title']}")
            print(f"{section['content']}")
            print(
                "========================================================================================================")
        print(f"标题数量: {title_count}")
        print(f"行数: {lines_count}")
        # 5 打印日志
        self._step_5_print_stats(lines_count, sections)
        # 6 备份
        self._step_6_backup(state, sections)

        state["chunks"] = None
        return state
    # 步骤一 参数处理
    def _step_1_get_inputs(self, state):
        print("node_document_split: 步骤1：参数处理")
        content = state.get("md_content")
        if not content:
            raise StateFieldError(field_name="content",expected_type=str)
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title",expected_type=str)
        # 换行处理
        content = content.replace("\r\n", " \n").replace("\r", " \n")

        return content,file_title

    # 步骤二 标题初切
    def _step_2_split_by_title(self, content, file_title):
        print("node_document_split: 步骤2：标题切(初切)")
        # 参数处理
        sections:List[Dict[str, str]] = []
        title_count:int = 0
        lines = content.split("\n")
        current_lines = []
        in_code_block = False
        current_title = ""

        # 切换逻辑(标题切)
        title_pattern = r'\s*#{1,6}\s+.+' # 标题正则

        # 独立封装刷新块的逻辑函数
        def _flush_section():
            # 没有内容时
            if not current_lines:
                return
            # 封装sections块
            sections.append({
                "file_title": "file_title",
                "title": current_title,
                "parent_title": "",
                "content": "\n".join(current_lines),

            })

        for line in lines:
            striped_line = line.strip()

            # 判断是否在代码块中
            """
                 .startswith（）判断字符串开头是不是指定的内容，返回True或者False
                 为True时进入代码块，为False时退出代码块
            """
            if striped_line.startswith("```")or striped_line.startswith("~~~"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            # 不在代码块中并且是标题的情况下
            """
                re.match(正则模板, 字符串)
                作用：从字符串最开头匹配正则表达式 title_pattern，匹配成功返回匹配对象，失败返回 None。
            """
            if not (in_code_block) and (re.match(title_pattern,line)):
                _flush_section()
                current_title = striped_line  # 换标题（把去掉特殊字符的标题换过来）
                current_lines = []
                # 先把新的标题放进缓存列表，后面循环读到的普通文字、代码块都会追加到这个列表里。
                current_lines = [current_title]
                # 标题计数器+1
                title_count+=1
            else :
                # 普通行或者代码块添加
                current_lines.append(line)

        # 兜底保存文档最后一章，避免末尾内容丢失
        _flush_section()
        return sections, title_count, len(lines)

    # 步骤三 无标题兜底
    def _step_3_handle_no_title(self, content, sections, title_count, file_title):
        print("node_document_split: 步骤3：无标题兜底(默认标题)")

        # 无标题时标题数量为0，直接写死
        if title_count == 0:
            return [{"title":"无标题","content":content,"file_title":file_title}]

        return sections

    def _step_4_refine_chunks(self, sections):
        print("node_document_split: 步骤4：块精细化处理(长切短合)")

        # 长切列表
        refined_split = []
        for sec in sections:
            refined_split.extend(self.split_long_section(sec)) # 长切操作

        # 短合列表
        final_sections = self.merge_short_sections(refined_split) # 短合操作

        # 给长切操作后的没有父标题的内容增加标题
        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""
        return final_sections

    def _step_5_print_stats(self, lines_count, sections):
        print("node_document_split: 步骤5：打印日志")
        pass

    def _step_6_backup(self, state, sections):
        print("node_document_split: 步骤6：备份")
        pass

    # 步骤四 方法一 长切操作
    def split_long_section(self, section:Dict[str,str]) -> List[Dict[str,str]]:
        print("node_document_split: 步骤4方法1长切")
        content = section.get("content","")
        content_len = len(content)

        # 判断长度是否符合最大字符要求
        # 长度在最大字符范围内直接返回
        if content_len <=get_config().max_content_length:
            return [section]

        title = section.get("title","") # 没有换行符的title
        prefix = f"{title}\n\n" if title else ""
        available_len = get_config().max_content_length - content_len # 切分标准

        # 去重标题
        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title)+len(title):].lstrip()

        # 切分器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len, # 切分标准
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "]
        )

        # 切分结果
        sub_sections = []
        # start=1：分片序号从1开始，不是0
        for index,chunk in enumerate(splitter.split(body),start=1):
            text = chunk.strip()
            # 空片段直接跳过，不生成数据
            if not text:
                continue
            # 拼接前缀+分片文本，再清除多余空格
            full_text = (prefix + text).strip()

            # 封装一条子分段数据
            sub_sections.append({
                "title": "",
                "content": full_text,
                "parent_title": title,
                "part":index,
                "file_title": section.get("file_title"),
            })

        return sub_sections

    def merge_short_sections(self, refined_split):
        print("node_document_split: 步骤4方法2短合")
        return refined_split


if __name__ == "__main__":
    node = NodeDocumentSplit()

    with open(r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\output\B530\B530_new.md","r",encoding="utf-8") as f:
        md_content = f.read()
    init_state = {
        "md_path": r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\output\B530\B530_new.md",
        "md_content": md_content,
        "file_title": "B530_new",
    }

    process = node.process(init_state)
    print(process)
