from __future__ import annotations

import difflib
import json
import logging
import os
import re

log = logging.getLogger(__name__)

SYNONYMS = {
    "txn": "transaction", "trx": "transaction", "trans": "transaction",
    "mail": "email", "cust": "customer", "client": "customer", "buyer": "customer",
    "amt": "amount", "total": "amount", "cost": "amount",
    "price": "amount", "value": "amount", "gross": "amount",
    "order": "purchase", "sale": "purchase", "sales": "purchase", "buy": "purchase",
    "dt": "date", "time": "date", "timestamp": "date", "ts": "date",
}


def _tokens(name: str) -> set[str]:
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    parts = re.split(r"[^a-zA-Z0-9]+", snake.lower())
    return {SYNONYMS.get(p, p) for p in parts if p}


def _token_weights(expected: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for col in expected:
        for tok in _tokens(col):
            counts[tok] = counts.get(tok, 0) + 1
    return {tok: 1.0 / n for tok, n in counts.items()}


def _score(actual: str, expected: str, weights: dict[str, float]) -> float:
    ta, te = _tokens(actual), _tokens(expected)
    total = sum(weights.get(t, 1.0) for t in te)
    shared = sum(weights.get(t, 1.0) for t in ta & te)
    overlap = shared / total if total else 0.0
    literal = difflib.SequenceMatcher(None, actual.lower(), expected.lower()).ratio()
    return 0.8 * overlap + 0.2 * literal


class RuleBasedHealer:
    name = "rule-based"

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def heal(self, expected: list[str], actual: list[str]) -> dict[str, str]:
        missing = [c for c in expected if c not in actual]
        unknown = [c for c in actual if c not in expected]
        weights = _token_weights(expected)

        pairs = sorted(
            ((_score(a, e, weights), a, e) for a in unknown for e in missing),
            key=lambda p: p[0],
            reverse=True,
        )

        mapping: dict[str, str] = {}
        taken: set[str] = set()
        for score, a, e in pairs:
            if score < self.threshold or a in mapping or e in taken:
                continue
            mapping[a] = e
            taken.add(e)
            log.info("  rename %s -> %s (rule)", a, e)
        return mapping


class OllamaHealer:
    def __init__(self, host: str, model: str, api_key: str = "", timeout: int = 300):
        self.host = host.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.fallback = RuleBasedHealer()
        self.disabled = False

    @property
    def name(self) -> str:
        return "ollama-cloud" if self.api_key else "ollama"

    @property
    def label(self) -> str:
        return f"{self.name}:{self.model}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _prompt(self, expected: list[str], actual: list[str]) -> str:
        return f"""You are a data engineer mapping incoming CSV columns onto a target database schema. Match them by meaning, not by spelling. Column names may be in any language.

Target schema columns: {expected}
Incoming CSV columns: {actual}

Reply with ONLY a JSON object of the form {{"incoming_column": "target_column"}}. Include only columns that need renaming. If an incoming column has no sensible match, leave it out. Never invent column names that are not in the two lists above."""

    def _ask(self, expected: list[str], actual: list[str]) -> dict:
        import requests

        resp = requests.post(
            f"{self.host}/api/chat",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": self._prompt(expected, actual)}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"])

    def heal(self, expected: list[str], actual: list[str]) -> dict[str, str]:
        import requests

        if self.disabled:
            return self.fallback.heal(expected, actual)

        try:
            raw = self._ask(expected, actual)
        except requests.ConnectionError:
            log.warning("  %s tidak merespons, lanjut rule-based", self.host)
            self.disabled = True
            return self.fallback.heal(expected, actual)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            log.error("  HTTP %d dari %s, lanjut rule-based", status, self.host)
            self.disabled = True
            return self.fallback.heal(expected, actual)
        except Exception as exc:
            log.warning("  LLM gagal (%s), lanjut rule-based", exc)
            return self.fallback.heal(expected, actual)

        if not isinstance(raw, dict):
            log.warning("  jawaban LLM bukan JSON, lanjut rule-based")
            return self.fallback.heal(expected, actual)

        mapping = {
            a: e for a, e in raw.items()
            if isinstance(e, str) and a in actual and e in expected and a not in expected
        }
        for a, e in mapping.items():
            log.info("  rename %s -> %s (llm)", a, e)
        if dropped := set(raw) - set(mapping):
            log.warning("  abaikan usulan LLM: %s", sorted(dropped))

        leftover = [c for c in expected if c not in mapping.values() and c not in actual]
        if leftover:
            renamed = [mapping.get(c, c) for c in actual]
            mapping.update(self.fallback.heal(expected, renamed))

        return mapping


def _ollama_config() -> tuple[str, str, str, int]:
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    default_host = "https://ollama.com" if api_key else "http://localhost:11434"
    default_model = "gpt-oss:20b" if api_key else "qwen3:4b"
    host = os.getenv("OLLAMA_HOST", default_host)
    model = os.getenv("OLLAMA_MODEL", default_model)
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))
    return host, model, api_key, timeout


def _ollama_alive(host: str, api_key: str = "", timeout: float = 5.0) -> bool:
    if api_key:
        return True
    try:
        import requests

        return requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout).ok
    except Exception:
        return False


def get_healer(engine: str = "auto"):
    host, model, api_key, timeout = _ollama_config()

    if engine == "rule":
        return RuleBasedHealer()
    if engine == "ollama":
        if not _ollama_alive(host, api_key):
            raise SystemExit(f"LLM tidak tersedia di {host}")
        return OllamaHealer(host, model, api_key, timeout)
    if _ollama_alive(host, api_key):
        return OllamaHealer(host, model, api_key, timeout)
    return RuleBasedHealer()
