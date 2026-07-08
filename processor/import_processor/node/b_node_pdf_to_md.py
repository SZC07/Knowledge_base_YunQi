import json
import logging
from pathlib import Path

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行...")

        # 1. 检查和获取相关参数
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)
        # 2.获取上传链接并且上传文件到mineru服务器
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)
        # 3.下载zip文件然后解压然后改名
        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)
        # 4.读取文件md_content
        with open(md_path, "r",encoding="utf-8") as md_file:
            md_content=md_file.read()

        # 5.设置state结果
        state["md_content"] = md_content
        state["md_path"] = md_path
        return state

    def _step_1_validate_paths(self, state: ImportGraphState):
        # 1.校验路径
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path",expected_type=str)
        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name="file_dir",expected_type=str)
        # 2.path封装路径
        pdf_path_obj = Path(pdf_path)
        output_dir_obj = Path(file_dir)
        # 3.文档是否存在
        if not pdf_path_obj.exists():
            raise StateFieldError(message="文档不存在：{pdf_path}")
        if not output_dir_obj.exists():
            raise StateFieldError(message="输出目录不存在{output_dir_path}")
        return pdf_path_obj, output_dir_obj

        return pdf_path_obj, output_dir_obj


    def _step_2_upload_and_poll(self, pdf_path_obj):
        logging.info(f"_step_2_upload_and_poll上传文件到mineru服务器...")
        return "下载url"

    def _step_3_download_and_extract(self, zip_url, output_dir_obj, stem):
        logging.info(f"_step_3_download_and_extract下载并解压改名")
        return "md_path"

if __name__ == "__main__":
    setup_logging()

    init_state={
        "pdf_path":r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\doc\H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf",
        "file_dir":r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\output",
    }
    node = NodePDFToMD()
    result = node.process(init_state)
    print(json.dumps(result,ensure_ascii=False, indent=4))

