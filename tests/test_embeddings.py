import dataclasses
import json
import math

import httpx
import pytest


def _settings(monkeypatch, **env):
    """Test-only settings copy, seen by both the embedding layer and the providers it builds on."""
    from app import embeddings, providers
    from app.config import settings
    vals = {"provider": "mock", "embed_provider": "", "embed_model": "", "embed_base_url": "", "embed_api_key": "",
            "embed_truncate": "auto", "embed_query_prefix": None, "embed_document_prefix": None, "embed_dim": 4,
            "llm_base_url": "https://chat.example.com/v1", "llm_api_key": "sk-chat", "llm_embed_model": "",
            "ollama_embed_model": "qwen3-embedding:0.6b", **env}
    s = dataclasses.replace(settings, **vals)
    monkeypatch.setattr(providers, "settings", s)
    monkeypatch.setattr(embeddings, "settings", s)
    return embeddings


def _capture(seen, dim=4):
    def handler(req):
        body = json.loads(req.content)
        seen.append({"url": str(req.url), "auth": req.headers.get("authorization"), **body})
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [1.0] * dim} for i in range(len(body["input"]))]})
    return handler


def _openai_embedder(e, handler):
    emb = e.get_embedder()
    emb.backend.client = httpx.Client(transport=httpx.MockTransport(handler))
    return emb


def test_follows_chat_provider_by_default(monkeypatch):
    e = _settings(monkeypatch, provider="ollama")
    assert e.embed_provider() == "ollama" and e.embed_model_label() == "qwen3-embedding:0.6b"
    e = _settings(monkeypatch, provider="bedrock", bedrock_embed_model="cohere.embed-v4:0")
    assert e.embed_model_label() == "cohere.embed-v4:0"
    e = _settings(monkeypatch, provider="foundry", embed_provider="ollama", embed_model="bge-m3")
    assert e.embed_provider() == "ollama" and e.embed_model_label() == "bge-m3"
    assert _settings(monkeypatch, embed_model="ignored").embed_model_label() == "mock-hash"


def test_openai_embeddings_on_another_service_never_get_the_chat_key(monkeypatch):
    seen = []
    e = _settings(monkeypatch, provider="foundry", embed_provider="openai", embed_model="text-embedding-3-large",
                  embed_base_url="https://api.openai.com/v1/embeddings", embed_api_key="sk-embed")
    _openai_embedder(e, _capture(seen)).embed(["x"])
    assert seen[0]["url"] == "https://api.openai.com/v1/embeddings" and seen[0]["auth"] == "Bearer sk-embed"
    assert seen[0]["model"] == "text-embedding-3-large" and seen[0]["dimensions"] == 4

    seen.clear()
    e = _settings(monkeypatch, provider="openai", embed_base_url="http://localhost:8080/v1", llm_embed_model="Qwen/Qwen3-Embedding-0.6B")
    _openai_embedder(e, _capture(seen)).embed(["x"])
    assert seen[0]["url"] == "http://localhost:8080/v1/embeddings" and seen[0]["auth"] is None


def test_openai_embeddings_on_the_chat_service_use_its_key(monkeypatch):
    seen = []
    e = _settings(monkeypatch, provider="openai", llm_embed_model="vendor/embed")
    _openai_embedder(e, _capture(seen)).embed(["x"])
    assert seen[0]["url"] == "https://chat.example.com/v1/embeddings" and seen[0]["auth"] == "Bearer sk-chat"

    seen.clear()
    e = _settings(monkeypatch, provider="foundry", embed_provider="openrouter", embed_model="qwen/qwen3-embedding-8b")
    _openai_embedder(e, _capture(seen)).embed(["x"])
    assert seen[0]["url"] == "https://openrouter.ai/api/v1/embeddings" and seen[0]["auth"] is None

    from app.providers import ProviderError
    with pytest.raises(ProviderError, match="EMBED_BASE_URL is empty"):
        _settings(monkeypatch, provider="foundry", embed_provider="openai").get_embedder()


def test_qwen3_gets_a_query_instruction_and_prefixes_can_be_overridden(monkeypatch):
    seen = []
    e = _settings(monkeypatch, provider="openrouter", llm_embed_model="qwen/qwen3-embedding-8b")
    emb = _openai_embedder(e, _capture(seen))
    emb.embed(["dry skin"], kind="query")
    emb.embed(["a toner"], kind="document")
    assert seen[0]["input"] == [f"Instruct: {e.QWEN3_TASK}\nQuery:dry skin"] and seen[1]["input"] == ["a toner"]

    assert e.preset_prefixes("intfloat/multilingual-e5-large") == ("query: ", "passage: ")
    assert e.preset_prefixes("text-embedding-3-large") == ("", "")
    seen.clear()
    e = _settings(monkeypatch, provider="openrouter", llm_embed_model="qwen/qwen3-embedding-8b",
                  embed_query_prefix="Find:\\n", embed_document_prefix="")
    emb = _openai_embedder(e, _capture(seen))
    emb.embed(["q"], kind="query")
    emb.embed(["d"])
    assert seen[0]["input"] == ["Find:\nq"] and seen[1]["input"] == ["d"]


def test_matryoshka_vectors_are_shortened_to_embed_dim(monkeypatch):
    seen = []
    e = _settings(monkeypatch, provider="openrouter", llm_embed_model="qwen/qwen3-embedding-8b")
    v = _openai_embedder(e, _capture(seen, dim=16)).embed(["x"])[0]      # service ignored `dimensions`
    assert len(v) == 4 and math.isclose(sum(x * x for x in v), 1.0)
    e = _settings(monkeypatch, provider="openrouter", llm_embed_model="vendor/fixed-size")
    assert len(_openai_embedder(e, _capture(seen, dim=16)).embed(["x"])[0]) == 16   # ingest reports the size
    e = _settings(monkeypatch, provider="openrouter", llm_embed_model="vendor/fixed-size", embed_truncate="true")
    assert len(_openai_embedder(e, _capture(seen, dim=16)).embed(["x"])[0]) == 4


class FakeEmbedder:
    """Stands in for a user's own embedding class (EMBED_PROVIDER=package.module:ClassName)."""
    embed_model = "default"

    def embed(self, texts, kind="document"):
        return [[float(len(t)), 0.0, 0.0, 1.0 if kind == "query" else 0.0] for t in texts]


def test_custom_class_and_chat_from_another_provider(monkeypatch):
    e = _settings(monkeypatch, embed_provider="tests.test_embeddings:FakeEmbedder", embed_model="my-model")
    assert e.embed_model_label() == "my-model"
    from app import providers
    p = providers.get_provider()
    assert p.name == "mock" and p.chat("s", "u") == ""
    assert p.embed(["abc"], kind="query") == [[3.0, 0.0, 0.0, 1.0]] and p.embedder.backend.embed_model == "my-model"

    from app.providers import ProviderError
    with pytest.raises(ProviderError, match="cannot load Nope"):
        _settings(monkeypatch, embed_provider="tests.test_embeddings:Nope").get_embedder()
