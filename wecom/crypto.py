"""企业微信消息加解密

基于企业微信官方加解密方案实现，用于回调消息的验签和解密。
参考: https://developer.work.weixin.qq.com/document/path/96238
"""

import base64
import hashlib
import socket
import struct
import time
import xml.etree.ElementTree as ET
from typing import Tuple

from Crypto.Cipher import AES


class WXBizMsgCrypt:
    """企业微信消息加解密类"""

    def __init__(self, token: str, encoding_aes_key: str, corp_id: str):
        self.token = token
        self.corp_id = corp_id
        self.aes_key = base64.b64decode(encoding_aes_key + "=")

    def verify_url(self, msg_signature: str, timestamp: str, nonce: str, echostr: str) -> str:
        """验证回调 URL 有效性 (GET 请求)

        Returns:
            解密后的 echostr 明文
        """
        self._check_signature(msg_signature, timestamp, nonce, echostr)
        return self._decrypt(echostr)

    def decrypt_msg(self, msg_signature: str, timestamp: str, nonce: str, post_data: str) -> str:
        """解密回调消息 (POST 请求)

        Args:
            post_data: POST 请求体 (XML 格式)

        Returns:
            解密后的 XML 明文
        """
        root = ET.fromstring(post_data)
        encrypt = root.find("Encrypt").text

        self._check_signature(msg_signature, timestamp, nonce, encrypt)
        return self._decrypt(encrypt)

    def encrypt_msg(self, reply_msg: str, nonce: str, timestamp: str = None) -> str:
        """加密回复消息

        Returns:
            加密后的 XML 响应
        """
        timestamp = timestamp or str(int(time.time()))
        encrypt = self._encrypt(reply_msg)
        signature = self._gen_signature(timestamp, nonce, encrypt)

        return f"""<xml>
<Encrypt><![CDATA[{encrypt}]]></Encrypt>
<MsgSignature><![CDATA[{signature}]]></MsgSignature>
<TimeStamp>{timestamp}</TimeStamp>
<Nonce><![CDATA[{nonce}]]></Nonce>
</xml>"""

    def _check_signature(self, msg_signature: str, timestamp: str, nonce: str, encrypt: str):
        """校验签名"""
        calculated = self._gen_signature(timestamp, nonce, encrypt)
        if calculated != msg_signature:
            raise ValueError(f"签名校验失败: expected={msg_signature}, got={calculated}")

    def _gen_signature(self, timestamp: str, nonce: str, encrypt: str) -> str:
        """生成签名"""
        sort_list = sorted([self.token, timestamp, nonce, encrypt])
        sha1 = hashlib.sha1("".join(sort_list).encode("utf-8")).hexdigest()
        return sha1

    def _decrypt(self, encrypted: str) -> str:
        """AES 解密"""
        cipher = AES.new(self.aes_key, AES.MODE_CBC, self.aes_key[:16])
        decrypted = cipher.decrypt(base64.b64decode(encrypted))

        # 去除 PKCS#7 padding
        pad_len = decrypted[-1]
        content = decrypted[:-pad_len]

        # 16字节随机串 + 4字节消息长度 + 消息明文 + corp_id
        msg_len = struct.unpack("!I", content[16:20])[0]
        msg = content[20:20 + msg_len].decode("utf-8")
        from_corp_id = content[20 + msg_len:].decode("utf-8")

        if from_corp_id != self.corp_id:
            raise ValueError(f"CorpID 不匹配: expected={self.corp_id}, got={from_corp_id}")

        return msg

    def _encrypt(self, text: str) -> str:
        """AES 加密"""
        # 16字节随机串
        random_str = self._get_random_str()
        text_bytes = text.encode("utf-8")
        # 拼接: 随机串 + 消息长度(网络字节序) + 消息明文 + corp_id
        content = (
            random_str.encode("utf-8")
            + struct.pack("!I", len(text_bytes))
            + text_bytes
            + self.corp_id.encode("utf-8")
        )

        # PKCS#7 padding
        block_size = 32
        pad_len = block_size - (len(content) % block_size)
        content += bytes([pad_len] * pad_len)

        cipher = AES.new(self.aes_key, AES.MODE_CBC, self.aes_key[:16])
        encrypted = cipher.encrypt(content)
        return base64.b64encode(encrypted).decode("utf-8")

    @staticmethod
    def _get_random_str() -> str:
        import random
        import string
        return "".join(random.choices(string.ascii_letters + string.digits, k=16))


def parse_xml_msg(xml_str: str) -> dict:
    """解析企微消息 XML 为字典"""
    root = ET.fromstring(xml_str)
    result = {}
    for child in root:
        result[child.tag] = child.text
    return result
