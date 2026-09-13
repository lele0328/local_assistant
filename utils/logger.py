"""
日志系统
"""
import os
from datetime import datetime


class Logger:
    """简单的日志记录器"""

    def __init__(self, log_path: str = "./data/logs"):
        self.log_path = log_path
        self._ensure_dir()

    def _ensure_dir(self):
        """确保日志目录存在"""
        os.makedirs(self.log_path, exist_ok=True)

    def _write(self, level: str, message: str):
        """写入日志"""
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{time}] [{level}] {message}\n"
        log_file = os.path.join(self.log_path, "app.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)

    def info(self, message: str):
        """记录信息"""
        self._write("INFO", message)

    def warning(self, message: str):
        """记录警告"""
        self._write("WARNING", message)

    def error(self, message: str):
        """记录错误"""
        self._write("ERROR", message)
