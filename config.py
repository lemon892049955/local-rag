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
for _subdir in ["topics", "entities", "concepts", "moc"]:
    (WIKI_DIR / _subdir).mkdir(exist_ok=True)

# ===== HTTP 代理 =====
HTTP_PROXY = os.getenv("HTTP_PROXY", "")  # 如 http://127.0.0.1:7890
HTTPS_PROXY = os.getenv("HTTPS_PROXY", "")  # 如 http://127.0.0.1:7890

def get_http_proxies() -> dict:
    """获取 HTTP 代理配置"""
    proxies = {}
    if HTTP_PROXY:
        proxies["http://"] = HTTP_PROXY
    if HTTPS_PROXY:
        proxies["https://"] = HTTPS_PROXY
    return proxies

def get_httpx_proxy() -> str:
    """获取 httpx 代理（单个 URL，用于 mounts）"""
    return HTTPS_PROXY or HTTP_PROXY or None

# ===== LLM Provider =====
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")

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


# ===== Vision API（独立配置） =====
def get_vision_config() -> dict:
    """获取 Vision API 配置（优先使用独立配置，否则复用 LLM 配置）"""
    # 优先使用独立的 Vision 配置
    vision_api_key = os.getenv("VISION_API_KEY", "")
    if vision_api_key:
        return {
            "api_key": vision_api_key,
            "base_url": os.getenv("VISION_BASE_URL", "https://api.openai.com/v1"),
            "model": os.getenv("VISION_MODEL", "gpt-4o-mini"),
        }

    # 否则复用 LLM 配置
    return get_llm_config()

# ===== Embedding =====
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
# bge 系列查询时需要加 instruction 前缀以区分 query/document，提升检索效果
EMBEDDING_QUERY_INSTRUCTION = os.getenv(
    "EMBEDDING_QUERY_INSTRUCTION",
    "为这个句子生成表示以用于检索相关文章：",
)

# ===== 服务 =====
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8900"))

# ===== API 认证 =====
API_TOKEN = os.getenv("API_TOKEN", "")  # 设置后，所有请求需带 Authorization: Bearer <token>
# 空字符串表示无认证（仅限本地开发）

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
