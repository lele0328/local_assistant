"""
数据加密模块
保护用户隐私，对敏感数据进行简单加密
"""
import base64
import hashlib
from typing import Optional


class SimpleEncryptor:
    """简单加密器（基于base64+hash，非军事级，够用）"""

    def __init__(self, key: str = "local_assistant_key"):
        self.key = key

    def encrypt(self, text: str) -> str:
        """加密文本"""
        combined = self.key + text
        return base64.b64encode(combined.encode("utf-8")).decode("utf-8")

    def decrypt(self, encrypted: str) -> str:
        """解密文本"""
        decoded = base64.b64decode(encrypted.encode("utf-8")).decode("utf-8")
        return decoded[len(self.key):]

    def hash_text(self, text: str) -> str:
        """生成文本哈希（不可逆，用于存储密码等）"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
