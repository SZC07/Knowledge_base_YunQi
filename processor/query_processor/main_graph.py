# 初始化环境变量（类加载前执行，保证全局生效）
from dotenv import load_dotenv

load_dotenv()


class KBQueryWorkflow:
    def __init__(self):
        pass