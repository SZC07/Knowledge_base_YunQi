import json
import logging
import time
import zipfile
from pathlib import Path

import requests

from config.mineru_config import mineru_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, PdfConversionError
from processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行...")

        # 1 检查和获取相关参数
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 2 获取上传链接并上传文件到mineru服务器
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)
        print(f"已经下载地址url:{zip_url}")

        # 3 下载zip压缩文件并且解压改名
        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        # 4 读取文件md_content
        with open(md_path, "r", encoding="utf-8") as md_file:
            md_content = md_file.read()

        # 5 设置state结果
        state["md_content"] = md_content
        state["md_path"] = md_path # E:\works-class\sz0525课程\掌柜智库课件0525\1.笔记\xxx.md
        return state

    def _step_1_validate_paths(self, state: ImportGraphState):
        """
        检查参数
        :param state:
        :return:
        """
        # 1 校验路径
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path", expected_type=str)
        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name="file_dir", expected_type=str)

        # 2 path封装路径
        pdf_path_obj = Path(pdf_path)
        output_dir_obj = Path(file_dir)

        # 3 文档是否存在
        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f"输入文件不存在: {pdf_path}")

        if not output_dir_obj.exists():
            raise FileProcessingError(message=f"输出目录不存在: {output_dir_obj}")

        return pdf_path_obj, output_dir_obj

    def _step_2_upload_and_poll(self, pdf_path_obj: Path):
        """
        上传并且获得下载链接
        :param pdf_path_obj:
        :return:
        """
        logging.info(f"_step_2_upload_and_poll上传文件到mineru服务器，并且获得下载地址...")

        # 1 校验api_token和base_url
        api_token = mineru_config.api_token
        base_url = mineru_config.base_url
        if not api_token:
            raise FileProcessingError(message="api_token未配置")
        if not base_url:
            raise FileProcessingError(message="base_url未配置")

        # 2 申请上传链接post
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }
        url = f"{base_url}/file-urls/batch"
        response = requests.post(url, headers=header, json=data)
        if response.status_code != 200:
            raise FileProcessingError(message=f"申请上传文件失败: {response.text}")
        result = response.json()
        if result.get("code") != 0:
            raise FileProcessingError(message=f"申请上传文件失败: {result.get('message')}")
        batch_id = result["data"]["batch_id"]
        signed_url = result["data"]["file_urls"][0]

        # 3 上传文件put
        with open(pdf_path_obj, "rb") as pdf_file:
            res_upload = requests.put(signed_url, data=pdf_file)
            if res_upload.status_code != 200:
                raise PdfConversionError(f"文件上传失败：状态码：{res_upload.status_code}，响应结果：{res_upload}")
            self.logger.info(f"文件上传成功！")

        # 4 获取下载链接get（循环+过期）
        poll_url = f"{base_url}/extract-results/batch/{batch_id}"  # 检查转化结果的接口
        start_time = time.time()  # 记录开始时间
        timeout_seconds = 600  # 最大超时时间
        poll_interval = 3  # 轮询间隔时间

        while True:
            end_time = time.time() - start_time
            if end_time > timeout_seconds:
                raise FileProcessingError(message="获得下载地址超时")
            try:
                res_poll = requests.get(url=poll_url, headers=header, timeout=10)  # 获得下载链接
            except Exception as e:
                self.logger.error(f"轮询接口异常：{e}")
                time.sleep(poll_interval)
                continue

            if res_poll.status_code != 200:
                raise PdfConversionError(f"【任务轮询】HTTP请求失败，状态码：{res_poll.status_code}，响应内容：{res_poll}")

            # 请求已经成功
            poll_data = res_poll.json()

            if poll_data.get("code") != 0:
                raise PdfConversionError(f"【任务轮询】任务失败，错误信息：{poll_data.get('message')}")

            extract_results = poll_data["data"]["extract_result"]  # 任务结果
            extract_result = extract_results[0]  # 下载链接对象
            extract_state = extract_result["state"]  # 下载状态
            print(f"转换结果:{extract_state}。。。。。。")
            if extract_state == "done":
                full_zip_url = extract_result["full_zip_url"]  # 获取下载链接
                return full_zip_url  # 返回下载链接
            elif extract_state == "failed":
                err_msg = extract_state.get("err_msg", "未知错误，无具体信息")
                raise PdfConversionError(f"【任务轮询】解析任务失败！batch_id：{batch_id}，错误信息：{err_msg}")
            else:
                self.logger.info(
                    f"【任务轮询】处理中... 已耗时{int(end_time)}s，状态：{extract_state}， batch_id：{batch_id}")
                time.sleep(poll_interval)

    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
        logging.info(f"_step_3_download_and_extract下载并解压改名")

        # 1 下载
        response = requests.get(zip_url)
        if response.status_code!=200:
            raise FileProcessingError(message=f"获得下载文件失败: {response.text}")
        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
        with open(zip_save_path,"wb") as f:
            f.write(response.content)

        # 2 创建目录
        extract_target_dir = output_dir_obj / pdf_stem
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        # 3 解压
        with zipfile.ZipFile(zip_save_path, "r") as  zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)
        self.logger.info(f"【ZIP解压】已清空旧的解压目录：{extract_target_dir}")

        # 4 改名
        self.logger.info(f"【MD重命名】找到MinerU生成的full.md文件")
        target_md_file = extract_target_dir / "full.md"
        self.logger.info(f"【MD重命名】开始将full.md文件进行重命名")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        target_md_file.rename(new_md_path)
        self.logger.info(f"【MD重命名】重命名成功，文件名：{pdf_stem}.md")

        return str(new_md_path.absolute()) # 返回绝对路径的字符串


if __name__ == "__main__":
    setup_logging()

    init_state = {
        "pdf_path": r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\doc\B530.pdf",
        "file_dir": r"E:\AI Sgg\3.阶段三\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\output"
    }

    node = NodePDFToMD()
    result = node(init_state)

    # 打印结果
    dumps = json.dumps(result, ensure_ascii=False, indent=4)

    print(dumps)