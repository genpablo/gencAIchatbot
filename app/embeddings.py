"""Embedding models, chosen independently of the chat model.

EMBED_PROVIDER picks the service (empty = the same as LLM_PROVIDER):
- ollama     : local, e.g. qwen3-embedding:0.6b (the default), bge-m3, nomic-embed-text
- openai     : any OpenAI-compatible /embeddings: OpenAI (text-embedding-3-*), vLLM, TEI, Infinity, LM Studio,
               Together, Mistral, Jina, Ollama's /v1 ... at EMBED_BASE_URL
- openrouter : the same, with OpenRouter's URL as the default (e.g. qwen/qwen3-embedding-8b)
- foundry    : a Microsoft Foundry deployment (e.g. text-embedding-3-large)
- bedrock    : Cohere Embed v3/v4 or Titan V2 on Amazon Bedrock
- mock       : hash vectors, for tests
- package.module:ClassName : your own class with embed(texts, kind) -> list of vectors, for anything else

EMBED_MODEL picks the model (empty = the service's own *_EMBED_* setting). The model's recommended
query / document prefixes are added for you (Qwen3-Embedding, E5, Nomic, mxbai), and vectors from Matryoshka
models are shortened to EMBED_DIM. Products embedded with another model are re-embedded by
`python -m app.ingest --embed` and left out of search until then.
"""
from __future__ import annotations

import importlib
import math
import re
from urllib.parse import urlsplit

from .config import settings
from .providers import (OPENAI_COMPATIBLE, OPENROUTER_URL, EmbedKind, OpenAICompatibleProvider, ProviderError,
                        openai_base, provider_for)

# What retrieval does, in the instruction format Qwen3-Embedding was trained with (queries only).
QWEN3_TASK = "Given a customer's skin profile and skincare needs, retrieve K-beauty products that suit them"

# (model id pattern, query prefix, document prefix), from each model's card
PREFIXES = [
    (r"qwen3-embedding", f"Instruct: {QWEN3_TASK}\nQuery:", ""),
    (r"(^|[/_-])e5-(small|base|large)", "query: ", "passage: "),
    (r"nomic-embed-text", "search_query: ", "search_document: "),
    (r"mxbai-embed-large", "Represent this sentence for searching relevant passages: ", ""),
]

# Models trained so that the first N values are themselves a good embedding (Matryoshka).
MATRYOSHKA = re.compile(r"qwen3-embedding|text-embedding-3|nomic-embed-text-v(1\.5|2)|mxbai-embed-large|embed-v4"
                        r"|jina-embeddings-v[34]|gemini-embedding|voyage-3")


def embed_provider() -> str:
    name = settings.embed_provider or settings.provider
    return name if ":" in name else name.lower()


def embed_model_label() -> str:
    """The embedding model's id, stored with every product vector (products.embedded_with)."""
    name = embed_provider()
    if name == "mock":
        return "mock-hash"
    if ":" in name:
        return settings.embed_model or name
    return settings.embed_model or {
        "bedrock": settings.bedrock_embed_model,
        "foundry": settings.foundry_embed_deployment, "azure": settings.foundry_embed_deployment,
        **dict.fromkeys(OPENAI_COMPATIBLE, settings.llm_embed_model),
    }.get(name, settings.ollama_embed_model)


class OpenAICompatibleEmbeddings(OpenAICompatibleProvider):
    """OpenAI-style /embeddings at EMBED_BASE_URL, or on the chat service when it is the same provider."""
    endpoint_var = "EMBED_BASE_URL"
    auth_hint = "Check EMBED_API_KEY."

    def _configure(self) -> None:
        name, own = embed_provider(), settings.embed_base_url
        same = name == settings.provider
        self.base = openai_base(own or (settings.llm_base_url if same else "")
                                or (OPENROUTER_URL if name == "openrouter" else ""), "EMBED_BASE_URL")
        # A separate embedding service never receives the chat key.
        self.key = settings.embed_api_key or ("" if own or not same else settings.llm_api_key)
        self.embed_model = settings.llm_embed_model
        self.timeout = settings.llm_timeout_s
        self.service = urlsplit(self.base).hostname or self.base


def _load_class(path: str):
    module, _, cls = path.partition(":")
    try:
        return getattr(importlib.import_module(module), cls)()
    except (ImportError, AttributeError) as e:
        raise ProviderError(f"EMBED_PROVIDER={path}: cannot load {cls} from {module} ({e}).") from e


def preset_prefixes(model: str) -> tuple[str, str]:
    for pattern, query, document in PREFIXES:
        if re.search(pattern, model.lower()):
            return query, document
    return "", ""


def shorten(vec: list[float], dim: int) -> list[float]:
    """First `dim` values of a Matryoshka embedding, scaled back to unit length."""
    if len(vec) <= dim:
        return vec
    vec = vec[:dim]
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


class Embedder:
    """An embedding service plus what the model needs: query/document prefixes and the EMBED_DIM size."""

    def __init__(self, backend, model: str):
        self.backend, self.model = backend, model
        query, document = preset_prefixes(model)
        own = {"query": settings.embed_query_prefix, "document": settings.embed_document_prefix}
        preset = {"query": query, "document": document}
        self.prefix = {k: preset[k] if own[k] is None else own[k].replace("\\n", "\n") for k in preset}
        self.truncate = settings.embed_truncate == "true" or (
            settings.embed_truncate == "auto" and bool(MATRYOSHKA.search(model.lower())))

    def embed(self, texts: list[str], kind: EmbedKind = "document") -> list[list[float]]:
        prefix = self.prefix[kind]
        vecs = self.backend.embed([prefix + t for t in texts] if prefix else texts, kind)
        return [shorten(v, settings.embed_dim) for v in vecs] if self.truncate else vecs


def get_embedder() -> Embedder:
    name, model = embed_provider(), embed_model_label()
    if ":" in name:
        backend = _load_class(name)
        if settings.embed_model:
            backend.embed_model = settings.embed_model
    else:
        backend = OpenAICompatibleEmbeddings() if name in OPENAI_COMPATIBLE else provider_for(name)
        backend.embed_model = model
    return Embedder(backend, model)
