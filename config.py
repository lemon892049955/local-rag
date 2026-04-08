"""全局配置 - 从环境变量加载，提供合理默认值"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ===== 路径 =====
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTORDB_DIR = BASE_DIR / "vectordb"
WIKI_DIR = BASE_DIR / "wiki"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
VECTORDB_DIR.mkdir(exist_ok=True)
WIKI_DIR.mkdir(exist_ok=True)
for _subdir in ["topics", "entities", "insights"]:
    (WIKI_DIR / _subdir).mkdir(exist_ok=True)

# ===== LLM Provider =====
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "kimi")

LLM_CONFIGS = {
    "kimi": {
        "api_key": os.getenv("KIMI_API_KEY", ""),
        "base_url": os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        "model": os.getenv("KIMI_MODEL", "moonshot-v1-8k"),
    },
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    },
    "ollama": {
        "api_key": "ollama",  # Ollama 不需要真实 key, 但 openai client 要求非空
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    },
}

def get_llm_config() -> dict:
    """获取当前 LLM 配置"""
    provider = LLM_PROVIDER.lower()
    if provider not in LLM_CONFIGS:
        raise ValueError(f"不支持的 LLM Provider: {provider}, 可选: {list(LLM_CONFIGS.keys())}")
    return LLM_CONFIGS[provider]

# ===== Embedding =====
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ===== 服务 =====
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8900"))

# ===== ChromaDB =====
CHROMA_COLLECTION_NAME = "knowledge_chunks"
WIKI_COLLECTION_NAME = "wiki_chunks"

# ===== 小红书 MCP =====
XHS_MCP_ENDPOINT = os.getenv("XHS_MCP_ENDPOINT", "")  # 如 http://localhost:18060/mcp

# ===== 企业微信 =====
def get_wecom_config() -> dict:
    """获取企业微信配置"""
    return {
        "corp_id": os.getenv("WECOM_CORP_ID", ""),          # 企业 ID
        "secret": os.getenv("WECOM_SECRET", ""),             # 应用 Secret
        "agent_id": int(os.getenv("WECOM_AGENT_ID", "0")),  # 应用 AgentID
        "token": os.getenv("WECOM_TOKEN", ""),               # 回调 Token
        "encoding_aes_key": os.getenv("WECOM_ENCODING_AES_KEY", ""),  # 回调 EncodingAESKey
    }
