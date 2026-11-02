"""Settings, read from environment variables or a .env file in the project root."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    load_dotenv(ROOT / ".env", override=False, interpolate=False)


_load_dotenv()


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    database_url: str = _env("DATABASE_URL", "postgresql://rag:rag@127.0.0.1:5432/kbeauty")

    # ollama (local testing) | bedrock (Amazon) | foundry (Microsoft Azure) | mock (no model, for CI)
    # openai (any OpenAI-compatible service) | openrouter (the same, OpenRouter's URL by default)
    provider: str = _env("LLM_PROVIDER", "ollama").lower()

    # --- Embeddings (independent of the chat model; see app/embeddings.py) ---
    # Empty = same service as LLM_PROVIDER. Or ollama | openai | openrouter | foundry | bedrock | mock,
    # or your own class as package.module:ClassName.
    embed_provider: str = _env("EMBED_PROVIDER", "").strip()
    # Empty = that service's own setting (OLLAMA_EMBED_MODEL, LLM_EMBED_MODEL, FOUNDRY_EMBED_DEPLOYMENT ...)
    embed_model: str = _env("EMBED_MODEL", "")
    # Only for openai/openrouter embeddings from another service than chat (the chat key is not sent there)
    embed_base_url: str = _env("EMBED_BASE_URL", "") or _env("LLM_EMBED_BASE_URL", "")
    embed_api_key: str = _env("EMBED_API_KEY", "") or _env("LLM_EMBED_API_KEY", "")
    # Every embedding model must output this size. 1024 works for Qwen3-Embedding-0.6B, bge-m3,
    # Cohere Embed v4, Titan V2, and Matryoshka models shortened to it (Qwen3-Embedding 4B/8B, text-embedding-3).
    embed_dim: int = int(_env("EMBED_DIM", "1024"))
    # Shorten longer vectors to EMBED_DIM: auto (only models trained for it) | true | false
    embed_truncate: str = _env("EMBED_TRUNCATE", "auto").lower()
    # Text put before queries / product texts. Unset = the model's recommended prefix, if it has one.
    embed_query_prefix: str | None = os.environ.get("EMBED_QUERY_PREFIX")
    embed_document_prefix: str | None = os.environ.get("EMBED_DOCUMENT_PREFIX")

    # Ask reasoning models (DeepSeek Flash, GPT Luna / GPT-5+, Claude Sonnet 5.5, Qwen3) not to think:
    # thinking tokens are billed as output and slow answers down, and these tasks do not need them.
    disable_thinking: bool = _env("DISABLE_THINKING", "true").lower() == "true"

    # --- Ollama (local) ---
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    ollama_chat_model: str = _env("OLLAMA_CHAT_MODEL", "qwen3.5:9b")
    ollama_parse_model: str = _env("OLLAMA_PARSE_MODEL", "qwen3.5:9b")
    ollama_embed_model: str = _env("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
    ollama_disable_thinking: bool = _env("OLLAMA_DISABLE_THINKING", _env("DISABLE_THINKING", "true")).lower() == "true"
    ollama_timeout_s: float = float(_env("OLLAMA_TIMEOUT_S", "300"))

    # --- Amazon Bedrock (production) ---
    aws_region: str = _env("AWS_REGION", "ap-northeast-2")
    bedrock_chat_model: str = _env("BEDROCK_CHAT_MODEL", "anthropic.claude-sonnet-5")
    bedrock_parse_model: str = _env("BEDROCK_PARSE_MODEL", "anthropic.claude-haiku-4-5-20251001-v1:0")
    bedrock_embed_model: str = _env("BEDROCK_EMBED_MODEL", "cohere.embed-v4:0")

    # --- Microsoft Foundry (Azure) ---
    # Endpoint from the Foundry portal, e.g. https://<resource>.openai.azure.com/openai/v1/
    foundry_endpoint: str = _env("FOUNDRY_ENDPOINT", "")
    foundry_auth: str = _env("FOUNDRY_AUTH", "key").lower()          # key | entra
    foundry_api_key: str = _env("FOUNDRY_API_KEY", "")
    foundry_chat_deployment: str = _env("FOUNDRY_CHAT_DEPLOYMENT", "gpt-4.1-mini")
    foundry_parse_deployment: str = _env("FOUNDRY_PARSE_DEPLOYMENT", "") or _env("FOUNDRY_CHAT_DEPLOYMENT", "gpt-4.1-mini")
    foundry_embed_deployment: str = _env("FOUNDRY_EMBED_DEPLOYMENT", "text-embedding-3-large")
    foundry_timeout_s: float = float(_env("FOUNDRY_TIMEOUT_S", "120"))

    # --- Any OpenAI-compatible service (OpenRouter, OpenAI, Together, vLLM ...) ---
    llm_base_url: str = _env("LLM_BASE_URL", "")
    llm_api_key: str = _env("LLM_API_KEY", "")                       # empty = no Authorization header (local servers)
    llm_chat_model: str = _env("LLM_CHAT_MODEL", "")
    llm_parse_model: str = _env("LLM_PARSE_MODEL", "") or _env("LLM_CHAT_MODEL", "")
    llm_embed_model: str = _env("LLM_EMBED_MODEL", "")             # e.g. qwen/qwen3-embedding-8b
    llm_timeout_s: float = float(_env("LLM_TIMEOUT_S", "120"))

    # Prices in the database are in this currency (used for budget parsing and display)
    currency: str = _env("CURRENCY", "USD").upper()
    # How to read ambiguous dates like 03/04/2027 in imported files: MDY (Mar 4) or DMY (3 Apr).
    # Unambiguous dates (2027-03-04, 31/03/2027, Excel date cells) are always read correctly.
    date_order: str = _env("DATE_ORDER", "MDY").upper()

    # --- Retrieval (ranking weights are set in the admin portal) ---
    retrieve_k: int = int(_env("RETRIEVE_K", "30"))
    shortlist_size: int = int(_env("SHORTLIST_SIZE", "20"))   # candidates the AI chooses options from

    # --- Admin portal (/admin) ---
    admin_password: str = _env("ADMIN_PASSWORD", "")          # empty = portal disabled
    secret_key: str = _env("SECRET_KEY", "")                  # signs the admin login cookie
    admin_session_hours: int = int(_env("ADMIN_SESSION_HOURS", "12"))

    # --- Staff chat app (/) ---
    staff_session_days: int = int(_env("STAFF_SESSION_DAYS", "30"))
    brand_name: str = _env("BRAND_NAME", "Gen C Beauty")
    # --- Customer assistant (/shop online, /store in store; set up in the admin portal) ---
    # Messages per visitor: every message may call the AI, and these pages need no login
    public_chat_per_minute: int = int(_env("PUBLIC_CHAT_PER_MINUTE", "8"))
    public_chat_per_hour: int = int(_env("PUBLIC_CHAT_PER_HOUR", "60"))
    # Sites allowed to show /shop in a frame (the BigCommerce chat bubble); space-separated, * wildcards allowed.
    # No other site may frame any page of this app.
    shop_embed_origins: str = _env("SHOP_EMBED_ORIGINS",
                                   "https://gencbeauty.com https://www.gencbeauty.com https://*.mybigcommerce.com")

    brand_logo_url: str = _env("BRAND_LOGO_URL", "https://cdn11.bigcommerce.com/s-kzkmuqjqk9/images/stencil/original/download1_1659212301__04676.original.png")


settings = Settings()
