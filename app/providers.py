"""LLM + embedding providers. Switch with LLM_PROVIDER = ollama | bedrock | foundry | openai | openrouter | mock.

- ollama : local testing (e.g. qwen3.5:9b for chat, bge-m3 for embeddings)
- foundry: production on Microsoft Foundry (Azure), OpenAI-compatible v1 endpoint: GPT, DeepSeek, Llama, Grok ...
           deployments; API key or Microsoft Entra ID
- bedrock: production on Amazon Bedrock (Claude / Qwen / DeepSeek via the Converse API,
           Cohere Embed v4 or Titan V2 for embeddings)
- openai : any third-party service with an OpenAI-compatible API (OpenAI, Together, Groq, vLLM, LM Studio ...);
           embeddings can come from a different service than chat
- openrouter: the same, with OpenRouter's URL as the default
- mock   : no model at all; deterministic hash embeddings, chat returns nothing so the
           recommender uses its rule-based fallback. Useful for CI and DB testing.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import Literal, Protocol
from urllib.parse import urlsplit

import httpx

from .config import settings

Role = Literal["parse", "rank"]
EmbedKind = Literal["document", "query"]


class Provider(Protocol):
    name: str

    def embed(self, texts: list[str], kind: EmbedKind = "document") -> list[list[float]]: ...
    def chat(self, system: str, user: str, role: Role = "rank") -> str: ...
    def model_label(self, role: Role) -> str: ...


# --------------------------------------------------------------------------- Ollama (local)
class ProviderError(RuntimeError):
    """A model call failed in a way the user can fix (service down, model not pulled, no access)."""


class OllamaProvider:
    name = "ollama"

    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(base_url=settings.ollama_url, timeout=settings.ollama_timeout_s)
        self._think_supported = True

    def model_label(self, role: Role) -> str:
        return settings.ollama_parse_model if role == "parse" else settings.ollama_chat_model

    def _post(self, path: str, body: dict) -> httpx.Response:
        try:
            return self.client.post(path, json=body)
        except httpx.ConnectError as e:
            raise ProviderError(f"Cannot reach Ollama at {settings.ollama_url}. Is Ollama running? "
                                f"(Windows: start the Ollama app, or run `ollama serve`)") from e
        except httpx.TimeoutException as e:
            raise ProviderError(f"Ollama timed out after {settings.ollama_timeout_s}s; the model may still be "
                                f"loading. Retry, or raise OLLAMA_TIMEOUT_S.") from e

    def _check(self, r: httpx.Response, model: str) -> None:
        if r.status_code == 404:
            raise ProviderError(f"Ollama model '{model}' is not installed. Run: ollama pull {model}")
        if r.status_code >= 400:
            raise ProviderError(f"Ollama error {r.status_code}: {r.text[:300]}")

    def embed(self, texts, kind="document"):
        out: list[list[float]] = []
        model = settings.ollama_embed_model
        for i in range(0, len(texts), 32):
            r = self._post("/api/embed", {"model": model, "input": texts[i:i + 32]})
            self._check(r, model)
            out.extend(r.json()["embeddings"])
        return out

    def chat(self, system, user, role="rank"):
        body = {
            "model": self.model_label(role),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2, "num_ctx": 8192},
        }
        if settings.ollama_disable_thinking and self._think_supported:
            body["think"] = False  # Qwen3 / Qwen3.5 answer much faster without the thinking pass
        r = self._post("/api/chat", body)
        if r.status_code == 400 and "think" in body:
            # model (or older Ollama) does not support the think flag -> retry without it
            self._think_supported = False
            body.pop("think")
            r = self._post("/api/chat", body)
        self._check(r, body["model"])
        return r.json()["message"]["content"]


# --------------------------------------------------------------------------- Amazon Bedrock
# Ways to switch Claude's thinking off, tried in order until the model accepts one: older Claude takes
# "disabled", Sonnet 5.5 only "between_tools", Opus 5.5 neither (it always thinks, so ask for low effort).
CLAUDE_THINKING_OFF = [{"thinking": {"type": "disabled"}}, {"thinking": {"type": "between_tools"}},
                       {"output_config": {"effort": "low"}}, None]


def _validation_message(e: Exception) -> str:
    """Lower-cased message of a Bedrock ValidationException (the model rejected a request field), else ''."""
    resp = getattr(e, "response", None)
    err = resp.get("Error", {}) if isinstance(resp, dict) else {}
    return (err.get("Message") or "").lower() if err.get("Code") == "ValidationException" else ""


class BedrockProvider:
    name = "bedrock"

    def __init__(self, client=None):
        if client is None:
            import boto3
            client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        self.rt = client
        self._opts: dict[str, dict] = {}            # per model: the request options it accepted

    def model_label(self, role: Role) -> str:
        return settings.bedrock_parse_model if role == "parse" else settings.bedrock_chat_model

    def embed(self, texts, kind="document"):
        model = settings.bedrock_embed_model
        if "titan-embed" in model:  # Titan V2: one text per call
            out = []
            for t in texts:
                body = {"inputText": t, "dimensions": settings.embed_dim, "normalize": True}
                resp = self.rt.invoke_model(modelId=model, body=json.dumps(body))
                out.append(json.loads(resp["body"].read())["embedding"])
            return out
        out = []  # Cohere Embed v3 / v4: up to 96 texts per call
        for i in range(0, len(texts), 96):
            body = {
                "texts": texts[i:i + 96],
                "input_type": "search_document" if kind == "document" else "search_query",
                "embedding_types": ["float"],
            }
            if "embed-v4" in model:
                body["output_dimension"] = settings.embed_dim
                body["truncate"] = "RIGHT"
            resp = self.rt.invoke_model(modelId=model, body=json.dumps(body))
            emb = json.loads(resp["body"].read())["embeddings"]
            out.extend(emb["float"] if isinstance(emb, dict) else emb)
        return out

    def chat(self, system, user, role="rank"):
        model = self.model_label(role)
        claude = settings.disable_thinking and ("anthropic" in model or "claude" in model)
        opts = self._opts.get(model) or {"temperature": True, "think": 0 if claude else len(CLAUDE_THINKING_OFF) - 1}
        while True:
            extra = CLAUDE_THINKING_OFF[opts["think"]]
            try:
                resp = self.rt.converse(
                    modelId=model,
                    system=[{"text": system}],
                    messages=[{"role": "user", "content": [{"text": user}]}],
                    inferenceConfig={"maxTokens": 1500, **({"temperature": 0.2} if opts["temperature"] else {})},
                    **({"additionalModelRequestFields": extra} if extra else {}),
                )
                break
            except Exception as e:
                msg = _validation_message(e)
                if opts["temperature"] and "temperature" in msg:
                    opts = {**opts, "temperature": False}       # Claude 4.7+ accept only the default sampling
                elif extra and any(w in msg for w in ("thinking", "effort", "output_config")):
                    opts = {**opts, "think": opts["think"] + 1}
                else:
                    raise
        self._opts[model] = opts
        parts = resp["output"]["message"]["content"]
        return "".join(p.get("text", "") for p in parts)


# --------------------------------------------------------------------------- OpenAI-compatible services
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENAI_COMPATIBLE = ("openai", "openrouter")


def openai_base(url: str) -> str:
    """Accepts a base URL or a pasted full endpoint and returns the base, e.g. https://openrouter.ai/api/v1"""
    u = (url or "").strip().split("?")[0].rstrip("/")
    if not u:
        raise ProviderError("LLM_BASE_URL is empty. Use your provider's OpenAI-compatible URL, "
                            f"e.g. {OPENROUTER_URL}")
    return re.sub(r"/(chat/completions|embeddings)$", "", u)


def foundry_base(endpoint: str) -> str:
    """Accepts what people paste from the portal and returns https://<host>/openai/v1"""
    e = (endpoint or "").strip().rstrip("/")
    if not e:
        raise ProviderError("FOUNDRY_ENDPOINT is empty. Copy it from the Foundry portal (Models + endpoints).")
    e = e.split("?")[0]
    if "/openai" in e:
        e = e[: e.index("/openai")]
    for suffix in ("/models", "/api/projects"):
        if suffix in e:
            e = e[: e.index(suffix)]
    return e + "/openai/v1"


REASONING_EFFORT_OFF = ["none", "minimal", "low"]     # lowest first; GPT-5 has "minimal" but not "none"


def relax_chat_options(opts: dict, error: str) -> dict:
    """Optional chat fields minus the ones a 400 error names (softened where possible); none named -> drop all."""
    error = error.lower()
    named = [k for k in opts if k in error]
    if not named:
        return {}
    out = dict(opts)
    for k in named:
        v = out.pop(k)
        if k == "max_tokens" and "max_completion_tokens" in error:
            out["max_completion_tokens"] = v                       # reasoning models renamed the limit
        elif k == "reasoning" and v["effort"] != REASONING_EFFORT_OFF[-1]:
            out[k] = {"effort": REASONING_EFFORT_OFF[-1]}          # model must think: keep it as short as possible
        elif (k == "reasoning_effort" and v in REASONING_EFFORT_OFF[:-1]
              and re.search(rf"\b{v}\b", error.replace(k, ""))):
            out[k] = REASONING_EFFORT_OFF[REASONING_EFFORT_OFF.index(v) + 1]   # the value, not the field, was rejected
    return out


class OpenAICompatibleProvider:
    """Any service with OpenAI-style /chat/completions and /embeddings: OpenRouter, OpenAI, Together, vLLM ..."""
    name = "openai"
    endpoint_var = "LLM_BASE_URL"
    auth_hint = "Check LLM_API_KEY."

    def __init__(self, client: httpx.Client | None = None, sleep=time.sleep):
        self._configure()
        self.client = client or httpx.Client(timeout=self.timeout)
        self.sleep = sleep
        self._send_dimensions = True
        self._chat_opts: dict[str, dict] = {}       # per model: the optional chat fields it accepted

    def _configure(self) -> None:
        self.base = openai_base(settings.llm_base_url or (OPENROUTER_URL if settings.provider == "openrouter" else ""))
        self.embed_base = openai_base(settings.llm_embed_base_url or self.base)
        self.timeout = settings.llm_timeout_s
        self.service = urlsplit(self.base).hostname or self.base

    def model_label(self, role: Role) -> str:
        return settings.llm_parse_model if role == "parse" else settings.llm_chat_model

    def _embed_model(self) -> str:
        return settings.llm_embed_model

    def _headers(self, embed: bool = False) -> dict:
        # A separate embedding service never receives the chat key.
        key = settings.llm_embed_api_key if embed and settings.llm_embed_base_url else settings.llm_api_key
        return {"Authorization": f"Bearer {key}"} if key else {}      # local servers need no key

    def _not_found(self, model: str) -> str:
        return f"Model '{model}' was not found at {self.service}. Check the model id and {self.endpoint_var}."

    def _thinking_off(self) -> dict:
        if "openrouter.ai" in self.base:
            return {"reasoning": {"effort": "none"}}                 # OpenRouter's one field for every model
        return {"reasoning_effort": REASONING_EFFORT_OFF[0],         # GPT-5+ / GPT Luna / o-series
                "thinking": {"type": "disabled"}}                    # DeepSeek V3.1+ / Flash

    def _post(self, url: str, body: dict, model: str, embed: bool = False) -> httpx.Response:
        for attempt in range(4):
            try:
                r = self.client.post(url, json=body, headers=self._headers(embed))
            except httpx.ConnectError as e:
                raise ProviderError(f"Cannot reach {self.service} at {url}. Check {self.endpoint_var} and the network.") from e
            except httpx.TimeoutException as e:
                raise ProviderError(f"{self.service} timed out after {self.timeout}s.") from e
            if (r.status_code in (408, 429) or r.status_code >= 500) and attempt < 3:
                wait = r.headers.get("retry-after-ms") or r.headers.get("retry-after")
                try:
                    secs = float(wait) / (1000 if "retry-after-ms" in r.headers else 1)
                except (TypeError, ValueError):
                    secs = 2 ** attempt
                self.sleep(min(max(secs, 0.5), 20))          # rate limit or temporary error: wait and retry
                continue
            break
        if r.status_code in (401, 403):
            raise ProviderError(f"{self.service} rejected the credentials ({r.status_code}). {self.auth_hint} {r.text[:300]}")
        if r.status_code == 402:
            raise ProviderError(f"{self.service} reports no credits left on this account or key (402).")
        if r.status_code == 404:
            raise ProviderError(self._not_found(model))
        if r.status_code == 429:
            raise ProviderError(f"{self.service} rate limit reached; try again shortly or raise the quota.")
        return r

    def _result(self, r: httpx.Response, field: str):
        """The reply's `field`. Gateways such as OpenRouter can answer 200 with only an error in the body."""
        try:
            data = r.json()
        except ValueError as e:
            raise ProviderError(f"{self.service} returned something that is not JSON: {r.text[:300]}") from e
        if not isinstance(data, dict) or not data.get(field):
            detail = data.get("error") or data if isinstance(data, dict) else data
            raise ProviderError(f"{self.service} returned no result: {str(detail)[:300]}")
        return data[field]

    def embed(self, texts, kind="document"):
        model, out = self._embed_model(), []
        if not model:
            raise ProviderError(f"No embedding model is set for {self.service}; check the model settings in .env.")
        url = f"{self.embed_base}/embeddings"
        for i in range(0, len(texts), 16):
            body = {"model": model, "input": texts[i:i + 16]}
            if self._send_dimensions:
                body["dimensions"] = settings.embed_dim     # text-embedding-3-* can shorten to EMBED_DIM
            r = self._post(url, body, model, embed=True)
            if r.status_code == 400 and "dimensions" in body:
                self._send_dimensions = False               # model has a fixed size; ingest checks EMBED_DIM
                body.pop("dimensions")
                r = self._post(url, body, model, embed=True)
            if r.status_code >= 400:
                raise ProviderError(f"{self.service} embeddings error {r.status_code}: {r.text[:300]}")
            out.extend(d["embedding"] for d in sorted(self._result(r, "data"), key=lambda d: d["index"]))
        return out

    def chat(self, system, user, role="rank"):
        model = self.model_label(role)
        if not model:
            raise ProviderError(f"No chat model is set for {self.service}; check the model settings in .env.")
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        opts = self._chat_opts.get(model)
        if opts is None:
            opts = {"temperature": 0.2, "max_tokens": 1500, "response_format": {"type": "json_object"}}
            if settings.disable_thinking:
                opts.update(self._thinking_off())
        for _ in range(8):
            r = self._post(f"{self.base}/chat/completions", {"model": model, "messages": msgs, **opts}, model)
            if r.status_code != 400 or not opts or "content_filter" in r.text:
                break
            opts = relax_chat_options(opts, r.text)     # the model rejected a field: drop or soften it
        if r.status_code >= 400:
            raise ProviderError(f"{self.service} chat error {r.status_code}: {r.text[:300]}")
        choices = self._result(r, "choices")
        self._chat_opts[model] = opts
        return choices[0]["message"].get("content") or ""


class FoundryProvider(OpenAICompatibleProvider):
    name = "foundry"
    endpoint_var = "FOUNDRY_ENDPOINT"
    auth_hint = "Check FOUNDRY_API_KEY, or the role assignment if you use Entra ID."

    def _configure(self) -> None:
        self.base = self.embed_base = foundry_base(settings.foundry_endpoint)
        self.timeout = settings.foundry_timeout_s
        self.service = "Microsoft Foundry"
        self._token, self._token_exp = None, 0.0

    def model_label(self, role: Role) -> str:
        return settings.foundry_parse_deployment if role == "parse" else settings.foundry_chat_deployment

    def _embed_model(self) -> str:
        return settings.foundry_embed_deployment

    def _headers(self, embed: bool = False) -> dict:
        if settings.foundry_auth == "entra":
            if not self._token or time.time() > self._token_exp - 300:
                try:
                    from azure.identity import DefaultAzureCredential
                except ImportError as e:
                    raise ProviderError("FOUNDRY_AUTH=entra needs: pip install azure-identity") from e
                tok = DefaultAzureCredential().get_token("https://ai.azure.com/.default")
                self._token, self._token_exp = tok.token, float(tok.expires_on)
            return {"Authorization": f"Bearer {self._token}"}
        if not settings.foundry_api_key:
            raise ProviderError("FOUNDRY_API_KEY is empty. Copy a key from the Foundry portal, or set FOUNDRY_AUTH=entra.")
        return {"api-key": settings.foundry_api_key}

    def _not_found(self, model: str) -> str:
        return (f"Foundry deployment '{model}' was not found. Use the deployment name from the "
                "Foundry portal (case-sensitive) and check FOUNDRY_ENDPOINT.")


# --------------------------------------------------------------------------- Mock (no model)
class MockProvider:
    name = "mock"

    def model_label(self, role: Role) -> str:
        return "mock"

    def embed(self, texts, kind="document"):
        return [_hash_embed(t, settings.embed_dim) for t in texts]

    def chat(self, system, user, role="rank"):
        return ""  # forces the deterministic fallback


def _hash_embed(text: str, dim: int) -> list[float]:
    """Bag of words hashed into `dim` buckets. Not semantic, but stable."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    v = [0.0] * dim
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 64) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def get_provider() -> Provider:
    if settings.provider == "bedrock":
        return BedrockProvider()
    if settings.provider in ("foundry", "azure"):
        return FoundryProvider()
    if settings.provider in OPENAI_COMPATIBLE:
        return OpenAICompatibleProvider()
    if settings.provider == "mock":
        return MockProvider()
    return OllamaProvider()


def embed_model_label() -> str:
    return {"bedrock": settings.bedrock_embed_model, "mock": "mock-hash",
            "foundry": settings.foundry_embed_deployment, "azure": settings.foundry_embed_deployment,
            **dict.fromkeys(OPENAI_COMPATIBLE, settings.llm_embed_model)}.get(
        settings.provider, settings.ollama_embed_model)


def extract_json(text: str) -> dict | None:
    """Parse a JSON object out of an LLM reply (handles <think> blocks and ```json fences)."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None
