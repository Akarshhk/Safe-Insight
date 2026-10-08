"""
Local LLM inference via llama-cpp-python.

Loads a GGUF model from disk and generates a **grounded** answer: the prompt
contains only the retrieved chunks, and the system instruction forbids using
anything else. No network, no API key, no telemetry.

Prompting strategy
------------------
Retrieved chunks are numbered ``[1] .. [k]`` and the model is told to cite those
numbers inline. Those numbers line up exactly with the citation array returned to
the UI (see :mod:`app.citations`), which is what turns "the model said so" into
"here is the paragraph it said it from".

The model is also told to answer *only* from the context and to say so when the
context is insufficient. On a 3B-class model this is the single highest-leverage
instruction: without it, small models confabulate freely.

Graceful degradation
--------------------
If ``llama-cpp-python`` is not installed or the GGUF file is missing, the module
does not crash the app. :func:`generate` returns a clearly-labelled placeholder
answer that still lists the retrieved chunks, so the retrieval and citation
pipeline can be demonstrated and tested end to end before the 2 GB model
download finishes. :func:`status` tells the UI which mode it is in.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from app import config

if TYPE_CHECKING:  # avoid importing numpy/faiss just to type-annotate.
    from app.vector_store import SearchResult

logger = logging.getLogger(__name__)

_llm = None                     # type: ignore[var-annotated]
_lock = threading.Lock()
#: llama.cpp contexts are not thread-safe and FastAPI runs ``/query`` on a
#: thread pool, so generations - and model swaps - take turns on this lock.
#: Lock order is always _generate_lock, then _lock.
_generate_lock = threading.Lock()
_load_error: Optional[str] = None

SYSTEM_PROMPT = (
    "You are Safe Insight, an offline document assistant for professionals who "
    "handle confidential material. Answer strictly and only from the numbered "
    "context passages provided by the user. Rules:\n"
    "1. Never use outside knowledge, and never guess.\n"
    "2. Cite the passage number in square brackets, e.g. [2], after every claim.\n"
    "3. If the passages do not contain the answer, reply exactly: "
    '"The provided documents do not contain enough information to answer that."\n'
    "4. Be concise and factual. Quote short phrases from the passages where "
    "precision matters."
)

GENERAL_SYSTEM_PROMPT = (
    "You are Safe Insight, an offline document assistant. "
    "If numbered context passages are provided, use them to answer the question, "
    "and cite the passage number in square brackets (e.g. [2]) after every claim. "
    "If the provided passages do not contain the answer, or if no passages are provided, "
    "use your general knowledge to answer conversationally. "
    "DO NOT cite any passage numbers if you are relying on general knowledge. "
    # Every generated token costs ~0.1 s on a laptop CPU, so length is latency.
    "Keep answers short and direct - a few sentences - unless the user asks "
    "for more detail."
)

#: The exact refusal string rule 3 asks for. The API surfaces this as
#: ``answered_from_context: false`` so the UI can style it differently.
INSUFFICIENT_CONTEXT_ANSWER = (
    "The provided documents do not contain enough information to answer that."
)

#: Appended when an answer ends early, so it never looks frozen mid-sentence.
ANSWER_CUT_OFF_NOTE = "\n\n[Answer cut off at the length limit.]"
ANSWER_STOPPED_NOTE = "\n\n[Stopped before the answer was finished.]"


@dataclass
class GenerationResult:
    """An answer plus the bookkeeping the audit log and UI need."""

    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    t_assembly: float = 0.0
    t_prefill: float = 0.0
    t_generation: float = 0.0
    stub: bool = False
    finish_reason: Optional[str] = None


class LLMUnavailable(RuntimeError):
    """Raised by :func:`get_llm` when no usable model can be loaded."""


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _load_llm():
    """
    Load the GGUF model with llama-cpp-python.

    ``n_ctx`` must be large enough for the system prompt + k chunks + the answer.
    With the default 500-token chunks and k=5 that is ~2 800 tokens of context,
    so 4 096 leaves comfortable headroom.
    """
    from llama_cpp import Llama  # imported lazily: pulls in the native library

    model_path = config.LLM_MODEL_PATH
    if not model_path.exists():
        raise LLMUnavailable(
            f"GGUF model not found at {model_path}. Run "
            f"'python download_model.py' once while online, or point "
            f"SAFE_INSIGHT_LLM_PATH at an existing .gguf file."
        )

    logger.info("Loading GGUF model %s (this takes 5-20 s on CPU)", model_path.name)
    return Llama(
        model_path=str(model_path),
        n_ctx=config.LLM_CONTEXT_TOKENS,
        n_threads=config.LLM_THREADS,
        n_gpu_layers=config.LLM_GPU_LAYERS,
        verbose=False,
        # Chat template comes from the GGUF metadata, so swapping Phi-3.5 for
        # Llama-3.2 needs no code change - only a different filename.
    )


def get_llm():
    """Return the loaded model, loading on first call (thread-safe)."""
    global _llm, _load_error
    if _llm is None:
        with _lock:
            if _llm is None:
                try:
                    _llm = _load_llm()
                    _load_error = None
                except Exception as exc:  # noqa: BLE001
                    _load_error = str(exc)
                    raise LLMUnavailable(_load_error) from exc
    return _llm


def use_model_file(filename: str) -> None:
    """Point config at a GGUF in the models dir; the next load picks it up."""
    config.LLM_MODEL_FILENAME = filename
    config.LLM_MODEL_PATH = config.MODELS_DIR / "llm" / filename
    config.LLM_CONTEXT_TOKENS = config._get_llm_context_tokens()


def reload_llm(filename: str) -> None:
    """Hot-swap the active model without restarting the backend."""
    global _llm, _load_error
    # Waiting for any in-flight answer also means the old model is freed before
    # the new one loads, rather than both sitting in RAM at once.
    with _generate_lock, _lock:
        if _llm is not None:
            # Drop the current reference; garbage collection frees the C structs.
            _llm = None

        use_model_file(filename)

        try:
            _llm = _load_llm()
            _load_error = None
        except Exception as exc:  # noqa: BLE001
            _load_error = str(exc)
            raise LLMUnavailable(_load_error) from exc


def is_loaded() -> bool:
    """True when the GGUF model is resident in memory."""
    return _llm is not None


def status() -> Dict[str, Any]:
    """Model status for ``GET /health`` and the UI's model indicator."""
    return {
        "loaded": is_loaded(),
        "model_path": str(config.LLM_MODEL_PATH),
        "model_file": config.LLM_MODEL_PATH.name,
        "model_present": config.LLM_MODEL_PATH.exists(),
        "context_tokens": config.LLM_CONTEXT_TOKENS,
        "threads": config.LLM_THREADS,
        "gpu_layers": config.LLM_GPU_LAYERS,
        "load_error": _load_error,
    }


def warm_up() -> None:
    """
    Load the model at startup so the first question is not 20 s slower.

    Failure is logged, not raised - the app still boots in stub mode and the UI
    shows why.
    """
    try:
        get_llm()
        logger.info("LLM ready: %s", config.LLM_MODEL_PATH.name)
    except LLMUnavailable as exc:
        logger.warning("LLM unavailable, running in stub mode: %s", exc)


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def build_context_block(results: Sequence[SearchResult]) -> str:
    """
    Render retrieved chunks as a numbered context block.

    The numbering is 1-based and matches ``Citation.index`` exactly, so a ``[3]``
    in the answer text resolves to the third citation card in the UI.
    """
    blocks: List[str] = []
    for position, result in enumerate(results, start=1):
        chunk = result.chunk
        location = (
            f"page {chunk.page_number}"
            if chunk.page_number is not None
            else f"chunk {chunk.chunk_index + 1}"
        )
        blocks.append(
            f"[{position}] Source: {chunk.filename} ({location})\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, results: Sequence[SearchResult]) -> str:
    """Assemble the user turn: context passages first, then the question."""
    if not results:
        return question
        
    return (
        "Context passages:\n"
        "-----------------\n"
        f"{build_context_block(results)}\n"
        "-----------------\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above, citing passage numbers in "
        "square brackets."
    )


def _stub_answer(question: str, results: Sequence[SearchResult]) -> GenerationResult:
    """
    Placeholder answer used when no GGUF model is available.

    Deliberately explicit that it is not a generated answer - a stub that looked
    like a real answer would be worse than no answer at all.
    """
    if not results:
        body = (
            "[Safe Insight is running without a language model.] "
            "No relevant passages were retrieved for this question."
        )
    else:
        preview = "\n".join(
            f"  [{position}] {r.chunk.filename} "
            f"({'page ' + str(r.chunk.page_number) if r.chunk.page_number else 'chunk ' + str(r.chunk.chunk_index + 1)}), "
            f"similarity {r.score:.3f}"
            for position, r in enumerate(results, start=1)
        )
        body = (
            "[Safe Insight is running without a language model - retrieval only.]\n"
            f"Retrieval found {len(results)} relevant passage(s) for your question:\n"
            f"{preview}\n\n"
            "Open the citations below to read the source text. To enable generated "
            "answers, run 'python download_model.py' and restart the backend."
        )
    return GenerationResult(
        answer=body,
        model="stub (no GGUF model loaded)",
        prompt_tokens=0,
        completion_tokens=0,
        stub=True,
        finish_reason="stub",
    )


def _count_tokens(llm: Any, *texts: str) -> int:
    """Token count by the loaded model's own tokenizer; 0 if it cannot tokenize."""
    try:
        return sum(len(llm.tokenize(text.encode("utf-8"), add_bos=False)) for text in texts if text)
    except Exception:  # noqa: BLE001 - bookkeeping must never fail a query
        return 0


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate(
    question: str,
    results: Sequence[SearchResult],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    strict_mode: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
    documents: Optional[Sequence[str]] = None,
    on_token: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> GenerationResult:
    """Thread-safe entry point; see :func:`_generate_unlocked` for the details."""
    with _generate_lock:
        return _generate_unlocked(
            question, results, max_tokens, temperature, strict_mode, history, documents,
            on_token, should_stop,
        )


def _generate_unlocked(
    question: str,
    results: Sequence[SearchResult],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    strict_mode: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
    documents: Optional[Sequence[str]] = None,
    on_token: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> GenerationResult:
    """
    Generate a grounded answer from the retrieved chunks.

    Args:
        question: the user's question, verbatim.
        results: retrieved chunks, best-first. If empty we short-circuit to the
            refusal string without spending inference time.
        max_tokens: output cap, defaults to ``config.LLM_MAX_OUTPUT_TOKENS``.
        temperature: defaults to ``config.LLM_TEMPERATURE`` (0.2 - low, because
            the job is faithful extraction, not creative writing).
        strict_mode: If False, falls back to general knowledge if context is absent.
        documents: filenames in the project, named in the system prompt so the
            model never claims nothing was uploaded.
        on_token: called with each piece of answer text as it is generated
            (streaming). The returned ``answer`` is still the complete text.
        should_stop: polled between pieces; returning True ends generation
            early and the answer is marked as stopped.

    Returns:
        A :class:`GenerationResult`. Never raises for a missing model; it falls
        back to :func:`_stub_answer` so the pipeline stays demonstrable.
    """
    if strict_mode and not results:
        return GenerationResult(
            answer=INSUFFICIENT_CONTEXT_ANSWER,
            model=config.LLM_MODEL_PATH.name,
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="no_context",
        )

    try:
        llm = get_llm()
    except LLMUnavailable:
        return _stub_answer(question, results)

    if should_stop is not None and should_stop():
        # Cancelled while waiting for the previous answer to finish.
        return GenerationResult(
            answer=ANSWER_STOPPED_NOTE.strip(),
            model=config.LLM_MODEL_PATH.name,
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="stopped",
        )

    t_assembly_start = time.perf_counter()
    prompt = build_prompt(question, results)
    sys_prompt = SYSTEM_PROMPT if strict_mode else GENERAL_SYSTEM_PROMPT
    if documents:
        # In the system prompt (not the question) so it stays a cached prefix.
        sys_prompt += "\n\nDocuments in this project: " + ", ".join(documents[:30]) + "."

    # Fast token estimation
    def approx_tokens(text: str) -> int:
        return len(text) // 4
        
    sys_tokens = approx_tokens(sys_prompt)
    prompt_tokens_est = approx_tokens(prompt)
    max_out = max_tokens or config.LLM_MAX_OUTPUT_TOKENS
    
    # Reserve tokens for current prompt, system prompt, and output.
    # History is re-read on every question, so it is hard-capped at ~500 tokens:
    # reading 1000 tokens alone took 10-20 s on a laptop CPU.
    available_history_tokens = min(
        500,
        config.LLM_CONTEXT_TOKENS - sys_tokens - prompt_tokens_est - max_out - 200
    )
    
    trimmed_history = []
    if history and available_history_tokens > 0:
        history_tokens = 0
        # Iterate from newest to oldest
        for msg in reversed(history):
            msg_tokens = approx_tokens(msg["content"])
            if history_tokens + msg_tokens > available_history_tokens:
                break
            history_tokens += msg_tokens
            trimmed_history.insert(0, msg)
            
    messages = [{"role": "system", "content": sys_prompt}]
    if trimmed_history:
        messages.extend(trimmed_history)
    messages.append({"role": "user", "content": prompt})
    t_assembly = time.perf_counter() - t_assembly_start
    
    t_prefill_start = time.perf_counter()
    t_prefill: Optional[float] = None
    content = ""
    finish_reason = None
    usage: Dict[str, Any] = {}
    try:
        response_stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_out,
            temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
            # Small models sometimes start echoing the passage list; these stop
            # sequences cut that off early rather than burning the token budget.
            stop=["\nContext passages:", "\nQuestion:"],
            stream=True,
        )
        try:
            for chunk in response_stream:
                if t_prefill is None:
                    # The first chunk arrives once the whole prompt has been read.
                    t_prefill = time.perf_counter() - t_prefill_start
                if should_stop is not None and should_stop():
                    finish_reason = "stopped"
                    break
                if chunk.get("choices"):
                    choice = chunk["choices"][0]
                    piece = choice["delta"].get("content", "")
                    if piece:
                        content += piece
                        if on_token is not None:
                            on_token(piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                if chunk.get("usage"):
                    usage = chunk["usage"]
        finally:
            # Ends llama.cpp's loop now rather than at garbage collection, so a
            # stopped answer frees the model for the next question immediately.
            response_stream.close()

    except Exception as exc:  # noqa: BLE001 - native errors surface as RuntimeError
        logger.exception("Generation failed")
        return GenerationResult(
            # Keep whatever the user already saw streamed, and say why it ended.
            answer=(
                f"{content.strip()}\n\n[Generation failed: {exc}]"
                if content.strip()
                else f"Generation failed locally: {exc}"
            ),
            model=config.LLM_MODEL_PATH.name,
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="error",
        )
    if t_prefill is None:
        t_prefill = time.perf_counter() - t_prefill_start
    t_generation = time.perf_counter() - t_prefill_start - t_prefill

    # llama.cpp's chat stream never reports usage, so count with the model's own
    # tokenizer. The prompt figure leaves out the chat template's few tokens.
    prompt_tokens = int(usage.get("prompt_tokens", 0)) or _count_tokens(
        llm, *(message["content"] for message in messages)
    )
    completion_tokens = int(usage.get("completion_tokens", 0)) or _count_tokens(llm, content)
    
    # Warn if we are approaching the context window limit
    if prompt_tokens > config.LLM_CONTEXT_TOKENS * 0.85:
        logger.warning(
            f"Prompt size ({prompt_tokens} tokens) is approaching the context limit "
            f"({config.LLM_CONTEXT_TOKENS} tokens). Consider reducing history or top_k."
        )

    # Say why an answer ended early, so it never looks like it froze mid-sentence.
    answer = content.strip()
    if finish_reason == "length":
        answer += ANSWER_CUT_OFF_NOTE
    elif finish_reason == "stopped":
        answer = (answer + ANSWER_STOPPED_NOTE).strip()
    return GenerationResult(
        answer=answer,
        model=config.LLM_MODEL_PATH.name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        t_assembly=t_assembly,
        t_prefill=t_prefill,
        t_generation=t_generation,
        finish_reason=finish_reason,
    )


__all__ = [
    "GenerationResult",
    "LLMUnavailable",
    "SYSTEM_PROMPT",
    "INSUFFICIENT_CONTEXT_ANSWER",
    "get_llm",
    "use_model_file",
    "reload_llm",
    "is_loaded",
    "status",
    "warm_up",
    "build_context_block",
    "build_prompt",
    "generate",
]
