"""
panelforge.model_adapters
=========================
Every judge agent and every policy model (the thing you're actually
fine-tuning) talks through a ModelAdapter. This is what makes the panel
provider-agnostic: an indie dev can mix Anthropic models, any OpenAI-compatible
endpoint (OpenAI itself, a local vLLM/Ollama/llama.cpp server, Together,
Groq, whatever exposes /chat/completions), and -- for testing without
spending a cent -- a deterministic MockAdapter.

Mixing providers is not a nicety, it's the main lever against correlated
judge bias: see PanelConfig.decorrelation_report and
safety_checks.agent_agreement_matrix.
"""
from __future__ import annotations

import hashlib
import os
import random
from abc import ABC, abstractmethod


class AdapterError(RuntimeError):
    pass


class ModelAdapter(ABC):
    """Minimal interface every backend must implement."""

    @abstractmethod
    async def complete(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        """Return the model's raw text completion for a single-turn exchange."""
        raise NotImplementedError


class AnthropicAdapter(ModelAdapter):
    """
    Talks to the Anthropic Messages API. Requires the `anthropic` package
    and an API key in the environment variable named by `api_key_env`
    (defaults to ANTHROPIC_API_KEY). Never bundle a real key in checked-in
    config -- api_key_env is a *name*, resolved at call time.
    """

    def __init__(self, model: str, api_key_env: str = "ANTHROPIC_API_KEY"):
        self.model = model
        self.api_key_env = api_key_env
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise AdapterError(
                    f"Environment variable {self.api_key_env} is not set. "
                    "Export your Anthropic API key before running the panel."
                )
            try:
                import anthropic  # local import: optional dependency
            except ImportError as e:
                raise AdapterError(
                    "The 'anthropic' package is required for AnthropicAdapter. "
                    "Install it with: pip install anthropic"
                ) from e
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._client

    async def complete(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        client = self._client_or_raise()
        resp = await client.messages.create(
            model=self.model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        parts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
        return "".join(parts)


class OpenAICompatAdapter(ModelAdapter):
    """
    Talks to any server implementing the OpenAI /chat/completions schema:
    OpenAI itself, vLLM, Ollama, LM Studio, Together, Groq, etc. This is
    usually how you'd add a *second* model family to decorrelate the panel
    from a single provider without paying for two hosted APIs -- point one
    agent at a local open-weights model.
    """

    def __init__(self, model: str, base_url: str, api_key_env: str = "OPENAI_API_KEY"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            try:
                import httpx  # local import: optional dependency
            except ImportError as e:
                raise AdapterError(
                    "The 'httpx' package is required for OpenAICompatAdapter. "
                    "Install it with: pip install httpx"
                ) from e
            api_key = os.environ.get(self.api_key_env, "")  # local servers often don't need one
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                timeout=60.0,
            )
        return self._client

    async def complete(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        client = self._client_or_raise()
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        r = await client.post("/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


class MockAdapter(ModelAdapter):
    """
    Deterministic, network-free stand-in used by the test suite and demo.py.
    It does NOT call any external model. Instead it derives a pseudo-opinion
    from a hash of (persona_name, prompt, response) so that:
      - the same input always gets the same verdict (reproducible tests)
      - different mock "personas" can be made to disagree on purpose, by
        biasing their hash function -- see `bias` below -- which is used in
        tests to simulate a sycophantic or length-biased judge and confirm
        safety_checks.py actually catches it.
    """

    def __init__(self, name: str, bias: str = "none", seed: int = 0):
        self.name = name
        self.bias = bias  # "none" | "prefers_longer" | "sycophantic" | "random"
        self.seed = seed

    async def complete(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        # This mock only knows how to answer the pairwise-judgment prompt
        # format produced by judge_agent.py; see _parse_pair_prompt below.
        prompt, resp_a, resp_b = _parse_pair_prompt(user)
        h = hashlib.sha256(f"{self.name}|{self.seed}|{prompt}|{resp_a}|{resp_b}".encode()).hexdigest()
        r = int(h[:8], 16) / 0xFFFFFFFF  # deterministic pseudo-random in [0, 1)

        if self.bias == "prefers_longer":
            winner = "A" if len(resp_a) >= len(resp_b) else "B"
            confidence = 0.9
            rationale = "Response chosen for being more thorough."  # a length-biased judge rationalizes, doesn't admit it
        elif self.bias == "sycophantic":
            # "sycophantic" mock: agrees with whichever response sounds more confident/agreeable
            winner = "A" if resp_a.count("!") + resp_a.lower().count("great") >= resp_b.count("!") + resp_b.lower().count("great") else "B"
            confidence = 0.85
            rationale = "Response chosen for its confident, agreeable tone."
        elif self.bias == "random":
            winner = random.Random(h).choice(["A", "B", "tie"])
            confidence = 0.5
            rationale = "No clear signal either way."
        else:
            winner = "A" if r < 0.5 else "B"
            if abs(r - 0.5) < 0.05:
                winner = "tie"
            confidence = round(0.5 + abs(r - 0.5), 2)
            rationale = f"Judged on substance per the '{self.name}' rubric."

        return f'{{"winner": "{winner}", "confidence": {confidence}, "rationale": "{rationale}"}}'


def _parse_pair_prompt(user: str) -> tuple[str, str, str]:
    """Extract (prompt, response_a, response_b) back out of the templated
    judge prompt built in judge_agent.py. Only used by MockAdapter."""
    def _between(text: str, start: str, end: str) -> str:
        i = text.find(start)
        j = text.find(end, i + len(start))
        if i == -1 or j == -1:
            return ""
        return text[i + len(start):j].strip()

    prompt = _between(user, "<prompt>", "</prompt>")
    resp_a = _between(user, "<response_a>", "</response_a>")
    resp_b = _between(user, "<response_b>", "</response_b>")
    return prompt, resp_a, resp_b


def build_adapter(spec) -> ModelAdapter:
    """Factory: turn an AgentSpec (config.py) into a live ModelAdapter."""
    if spec.provider == "anthropic":
        return AnthropicAdapter(model=spec.model, api_key_env=spec.api_key_env or "ANTHROPIC_API_KEY")
    if spec.provider == "openai_compatible":
        if not spec.base_url:
            raise AdapterError(f"Agent '{spec.id}' uses provider=openai_compatible but has no base_url set.")
        return OpenAICompatAdapter(model=spec.model, base_url=spec.base_url, api_key_env=spec.api_key_env or "OPENAI_API_KEY")
    if spec.provider == "mock":
        # model field doubles as the bias name for mocks, e.g. model: "prefers_longer"
        return MockAdapter(name=spec.persona.name, bias=spec.model)
    raise AdapterError(f"Unknown provider '{spec.provider}' for agent '{spec.id}'.")
