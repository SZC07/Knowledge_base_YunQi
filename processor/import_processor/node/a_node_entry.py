import logging
from pathlib import Path

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState


class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    """

    name = "node_entry"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行...")

        # 1.获得输入路径
        import_file_path = state.get("import_file_path")
        # 2.校验路径
        if not import_file_path:
            raise StateFieldError(field_name="import_file_path",expected_type=str)
        # 3. 4. 校验文件
        import_file_path_obj = Path(import_file_path)
        # .exists()：判断当前路径是否真实存在（文件 / 文件夹都适用）
        if not import_file_path_obj.exists():
            raise StateFieldError(message=f"文件不存在:{import_file_path}")
        # 5. md还是pdf
        # .suffix 文件后缀名
        if import_file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path

        elif import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path

        else:
            raise FileProcessingError(message="不支持的文件格式:{import_file_path}")

        # .stem：纯文件名（无后缀）
        state["file_title"] = import_file_path_obj.stem
        state["file_dir"] = r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\output"
        return state