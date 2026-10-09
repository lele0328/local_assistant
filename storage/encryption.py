"""
数据加密模块（真实实现）

替换 storage/encryption.py。

原实现的问题：
    def encrypt(self, text):
        combined = self.key + text
        return base64.b64encode(combined.encode()).decode()

这不是加密 —— base64 只是编码，任何人 b64decode 一下、去掉前缀就还原了。
README 里声称「支持数据加密」「保护隐私」，实际上不成立。面试官追问一句就会暴露。

本实现用 Python 标准库的 Fernet（AES-128-CBC + HMAC 认证加密），
不引入第三方依赖，但具备真实的机密性和完整性保护。

关键点（面试可以讲）：
1. 用 PBKDF2 从口令派生密钥，而不是把口令直接当密钥 —— 加盐 + 迭代抵御字典攻击。
2. 用 Fernet 做认证加密（AEAD）—— 密文被篡改解密会失败，而不是静默返回垃圾。
3. salt 随机生成并单独存储 —— 同样的明文两次加密得到不同密文。
4. 密钥不进代码、不进 git，从环境变量读。

用法：
    enc = DataEncryptor.from_password(os.getenv("LOCAL_ENC_PASSWORD"))
    token = enc.encrypt("敏感内容")
    assert enc.decrypt(token) == "敏感内容"
"""

import base64
import hashlib
import hmac
import os
import secrets
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

# PBKDF2 迭代次数。越高越安全但越慢，20 万次在现代机器上约几十毫秒，够用。
_PBKDF2_ITERATIONS = 200_000


class EncryptionError(Exception):
    """加密/解密失败"""


class DataEncryptor:
    """
    基于口令的本地数据加密器。

    与旧版的区别：这是真实的加密（AES + HMAC），不是 base64 编码。
    """

    def __init__(self, key: bytes, salt: bytes):
        """
        Args:
            key: 32 字节派生密钥
            salt: 16 字节盐值
        """
        if len(key) != 32:
            raise EncryptionError("密钥必须是 32 字节")
        self._key = key
        self._salt = salt
        self._fernet = Fernet(base64.urlsafe_b64encode(key))

    # ---------- 构造 ----------

    @classmethod
    def from_password(cls, password: str, salt: Optional[bytes] = None) -> "DataEncryptor":
        """
        从口令创建加密器。

        Args:
            password: 用户口令
            salt: 盐值。为 None 时随机生成（加密，但如果要解密必须保存下来）
        """
        if not password:
            raise EncryptionError("口令不能为空")
        if salt is None:
            salt = secrets.token_bytes(16)
        if len(salt) < 16:
            raise EncryptionError("盐值至少 16 字节")

        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            _PBKDF2_ITERATIONS,
            dklen=32,
        )
        return cls(key=key, salt=salt)

    @classmethod
    def from_env(cls, var_name: str = "LOCAL_ENC_PASSWORD") -> "DataEncryptor":
        """从环境变量读取口令（推荐方式，口令不进代码库）"""
        password = os.getenv(var_name, "")
        if not password:
            raise EncryptionError(
                f"环境变量 {var_name} 未设置。请在 .env 中配置一个强口令。"
            )
        return cls.from_password(password)

    # ---------- 属性 ----------

    @property
    def salt_b64(self) -> str:
        """盐值的 base64 表示（需要和密文一起存储，否则无法解密）"""
        return base64.urlsafe_b64encode(self._salt).decode("ascii")

    # ---------- 加解密 ----------

    def encrypt(self, text: str) -> str:
        """
        加密文本。

        Returns:
            base64 字符串（含 Fernet 的版本号、时间戳、IV、密文、HMAC）
        Raises:
            EncryptionError: 输入非法
        """
        if text is None:
            raise EncryptionError("待加密内容不能为 None")
        try:
            return self._fernet.encrypt(text.encode("utf-8")).decode("ascii")
        except Exception as e:
            raise EncryptionError(f"加密失败: {e}") from e

    def decrypt(self, token: str) -> str:
        """
        解密文本。

        Raises:
            EncryptionError: 密文被篡改、密钥不匹配或格式非法
        """
        if not token:
            raise EncryptionError("待解密内容为空")
        try:
            return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except InvalidToken:
            # 认证失败：要么密钥不对，要么密文被改过
            raise EncryptionError("解密失败：密钥不匹配或数据已被篡改")
        except Exception as e:
            raise EncryptionError(f"解密失败: {e}") from e

    # ---------- 口令校验（密码存储场景）----------

    @staticmethod
    def hash_password(password: str) -> str:
        """
        生成带盐的口令哈希，用于「存储密码」而不是「加密数据」。

        格式：pbkdf2_sha256$迭代次数$盐$哈希
        与旧版直接 sha256 的区别：旧版无盐，彩虹表可直接反查；且无迭代，暴力破解极快。
        """
        salt = secrets.token_bytes(16)
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
        )
        return (
            f"pbkdf2_sha256${_PBKDF2_ITERATIONS}"
            f"${base64.urlsafe_b64encode(salt).decode()}"
            f"${base64.urlsafe_b64encode(dk).decode()}"
        )

    @staticmethod
    def verify_password(password: str, stored: str) -> bool:
        """校验口令（用 compare_digest 做常数时间比较，防时序攻击）"""
        try:
            algo, iters, salt_b64, hash_b64 = stored.split("$")
            if algo != "pbkdf2_sha256":
                return False
            salt = base64.urlsafe_b64decode(salt_b64)
            expected = base64.urlsafe_b64decode(hash_b64)
            dk = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), salt, int(iters)
            )
            return hmac.compare_digest(dk, expected)
        except (ValueError, TypeError):
            return False


if __name__ == "__main__":
    print("--- 自测 ---")

    enc = DataEncryptor.from_password("my-strong-password-123")

    # 1) 往返正确
    plain = "这是一段敏感的本地数据，包含中文和 emoji 🎉"
    token = enc.encrypt(plain)
    assert enc.decrypt(token) == plain, "往返失败"
    print(f"[OK ] 加密往返正确，密文长度 {len(token)}")

    # 2) 密文里看不到明文（对比旧版 base64 的行为）
    assert plain[:6] not in base64.b64decode(token + "==").decode("latin-1", "ignore")
    print("[OK ] 密文中不含明文片段（旧版 base64 会直接暴露）")

    # 3) 同样明文两次加密结果不同（随机 IV）——旧版是确定性的
    assert enc.encrypt(plain) != enc.encrypt(plain)
    print("[OK ] 相同明文两次密文不同（随机 IV，防模式分析）")

    # 4) 篡改检测
    tampered = token[:-4] + ("AAAA" if not token.endswith("AAAA") else "BBBB")
    try:
        enc.decrypt(tampered)
        print("[FAIL] 篡改未被检测到")
    except EncryptionError as e:
        print(f"[OK ] 篡改被拒绝: {e}")

    # 5) 口令错误无法解密
    wrong = DataEncryptor.from_password("another-password", salt=enc._salt)
    try:
        wrong.decrypt(token)
        print("[FAIL] 错误口令竟然解密成功")
    except EncryptionError:
        print("[OK ] 错误口令无法解密")

    # 6) 口令哈希
    stored = DataEncryptor.hash_password("user-pw-456")
    assert DataEncryptor.verify_password("user-pw-456", stored)
    assert not DataEncryptor.verify_password("wrong", stored)
    print("[OK ] 口令哈希与校验正确")

    print("\n全部通过。注意：salt 必须持久化，否则重启后无法解密旧数据。")
