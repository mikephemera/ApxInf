"""Prompt construction and tokenisation.

The reference uses the SentencePiece *Python* package rather than the engine's
native binding: the reference has to run on any device from a plain virtual
environment, and `apxinf_py` is a CUDA-only extension. Both wrap the same C++
library, and the repository already treats the Python package as the
differential reference for the binding (`python/apxinf/tests/test_tokenize.py`).

The prompt string is not free-form. It decides the token ids, which decide the
whole rollout, so the construction is reused from `python/apxinf` rather than
restated:

* the task text is cleaned with ``strip`` and ``_``/newline replaced by spaces;
* with ``discrete_state`` the proprioception is discretised into the
  ``-1..=255`` bins and spliced in as ``Task: ..., State: ...;\\nAction: ``;
* the encoding appends a BOS, and a *separately encoded* newline token when the
  state is not being spliced in -- not a newline appended to the text, which
  tokenises differently;
* the result must be ``1..=max_token_len`` tokens.

The discretisation is imported rather than reimplemented. `processors/
tokenize.py:34-66` documents two traps that a reimplementation walks into: the
signed underflow bin ``-1`` must survive because openpi writes it verbatim, and
``floor((v + 1) * 128)`` is not equivalent to ``digitize`` at a bin edge.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

__all__ = [
    "DEFAULT_MAX_TOKEN_LEN",
    "PromptTokenizer",
    "SyntheticTokenizer",
    "encode_prompt",
    "sha256_of",
]

DEFAULT_MAX_TOKEN_LEN = 200


def sha256_of(path: str | Path) -> str:
    """Digest of a tokenizer file, for pinning which one a run used.

    `python/apxinf/apxinf/checkpoints/layout.py:453` warns that the two sides of
    a comparison must use the *same* file or their token ids are not comparable.
    A digest is the only way to know that after the fact.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _discretize_state(state: Sequence[float]):
    try:
        from apxinf.processors.tokenize import discretize_state  # type: ignore
    except ImportError as error:  # pragma: no cover - depends on the host
        raise ImportError(
            "discrete-state prompts need `apxinf.processors.tokenize."
            "discretize_state` from python/apxinf; it is imported rather than "
            "reimplemented because its edge-case behaviour is load-bearing."
        ) from error
    return discretize_state(state)


def _build_prompt(prompt: str, state, *, discrete_state: bool) -> str:
    try:
        from apxinf.processors.tokenize import build_prompt  # type: ignore
    except ImportError as error:  # pragma: no cover - depends on the host
        raise ImportError(
            "prompt construction needs `apxinf.processors.tokenize.build_prompt` "
            "from python/apxinf, which is kept in step with the Rust "
            "`pi05_prompt`."
        ) from error
    return build_prompt(prompt, state=state, discrete_state=discrete_state)


class PromptTokenizer:
    """Turns a task string (and optionally a state) into ``uint32`` token ids."""

    PARAMS = ("max_token_len", "discrete_state")

    def __init__(
        self,
        model_path: str | Path,
        max_token_len: int = DEFAULT_MAX_TOKEN_LEN,
        discrete_state: bool = True,
    ) -> None:
        import sentencepiece

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"no SentencePiece model at {self.model_path}. PI0.5's tokenizer "
                "is not distributed with the checkpoint; `layout.py:422-455` "
                "resolves it from APXINF_TOKENIZER or the model directory, and "
                "the two sides of a comparison must use the same file."
            )
        self.digest = sha256_of(self.model_path)
        self.max_token_len = int(max_token_len)
        self.discrete_state = bool(discrete_state)
        self._processor = sentencepiece.SentencePieceProcessor(model_file=str(self.model_path))

    def _encode(self, text: str):
        return [int(token) for token in self._processor.encode(text, add_bos=True, add_eos=False)]

    def text(self, prompt: str, state: Optional[Sequence[float]] = None) -> str:
        """The exact string this tokenizer encodes for ``prompt`` and ``state``.

        Exposed because a caller that records the prompt template's output -- a
        capture writes it into the bundle -- must record the string the encoder
        actually saw. Calling ``build_prompt`` again with the same arguments
        would produce the same characters today and is exactly the kind of
        second derivation that drifts once one side gains a parameter.
        """
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt)!r}")
        return _build_prompt(prompt, state, discrete_state=self.discrete_state)

    def __call__(self, prompt: str, state: Optional[Sequence[float]] = None):
        import numpy as np

        tokens = self._encode(self.text(prompt, state))
        if not self.discrete_state:
            # The upstream reference appends a standalone newline *token*, which
            # is not the same as encoding a newline appended to the text.
            tokens = tokens + self._encode("\n")
        if not 0 < len(tokens) <= self.max_token_len:
            raise ValueError(
                f"token count must be in 1..={self.max_token_len}, got {len(tokens)}"
            )
        return np.asarray(tokens, dtype=np.uint32)


class SyntheticTokenizer:
    """A checkpoint-free stand-in: a deterministic ramp of ids.

    Carries no meaning and is for latency work with random weights only. It
    exists so that a run which never needed a tokenizer cannot accidentally
    depend on one being present.
    """

    PARAMS = ("token_count", "max_token_len")
    discrete_state = False

    def __init__(self, token_count: int, max_token_len: int = DEFAULT_MAX_TOKEN_LEN) -> None:
        self.token_count = int(token_count)
        self.max_token_len = int(max_token_len)
        if not 0 < self.token_count <= self.max_token_len:
            raise ValueError(
                f"token_count must be in 1..={self.max_token_len}, got {self.token_count}"
            )

    def __call__(self, prompt: str, state: Optional[Sequence[float]] = None):
        import numpy as np

        del prompt, state
        return (np.arange(self.token_count, dtype=np.uint32) % np.uint32(256)) + np.uint32(1)


def encode_prompt(
    prompt: str,
    tokenizer_path: str | Path,
    *,
    state: Optional[Sequence[float]] = None,
    discrete_state: bool = True,
    max_token_len: int = DEFAULT_MAX_TOKEN_LEN,
):
    """One-shot convenience wrapper around :class:`PromptTokenizer`."""
    return PromptTokenizer(
        tokenizer_path, max_token_len=max_token_len, discrete_state=discrete_state
    )(prompt, state)
