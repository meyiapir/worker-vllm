"""Minimal RunPod serverless handler for GLM-5.2 on vLLM 0.23.x.

Depends ONLY on vLLM's stable top-level API (`vllm.LLM`, `vllm.SamplingParams`),
never `vllm.entrypoints.*` (whose layout churns across minors). For TP=8 vLLM
uses the `spawn` start method, so child processes RE-IMPORT this module — every
side effect that starts processes (the LLM engine) or the runpod worker MUST be
guarded so children don't re-run it:
  * the engine is created lazily inside the handler (main process only, on the
    first job), so a re-import never touches it;
  * `runpod.serverless.start` is under `if __name__ == "__main__"`.
Lazy load also lets the worker report ready immediately and do the ~755GB pull
during the first job (covered by the execution timeout) instead of blocking boot.
"""
import os
import threading
import time

import runpod
from vllm import LLM, SamplingParams


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


MODEL = os.environ.get("MODEL_NAME", "zai-org/GLM-5.2-FP8")

_llm = None
_load_seconds = None
_lock = threading.Lock()


def _get_llm():
    """Build the engine once, in the main process, on first use."""
    global _llm, _load_seconds
    if _llm is None:
        with _lock:
            if _llm is None:
                t0 = time.time()
                engine = LLM(
                    model=MODEL,
                    tensor_parallel_size=_int("TENSOR_PARALLEL_SIZE", 8),
                    kv_cache_dtype=os.environ.get("KV_CACHE_DTYPE", "fp8"),
                    max_model_len=_int("MAX_MODEL_LEN", 262144),
                    gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9")),
                    trust_remote_code=True,
                )
                _load_seconds = round(time.time() - t0, 1)
                print(f"[handler] model loaded in {_load_seconds}s: {MODEL}", flush=True)
                _llm = engine
    return _llm


def handler(job):
    inp = job.get("input", {}) or {}
    sp = SamplingParams(
        temperature=float(inp.get("temperature", 0.7)),
        top_p=float(inp.get("top_p", 0.95)),
        max_tokens=_int_val(inp.get("max_tokens"), 512),
    )
    llm = _get_llm()  # first job blocks here while the model is pulled + sharded
    t0 = time.time()
    if inp.get("messages"):
        outs = llm.chat(messages=inp["messages"], sampling_params=sp)
    else:
        outs = llm.generate(inp.get("prompt", "Hello"), sampling_params=sp)
    dt = time.time() - t0

    out = outs[0].outputs[0]
    n_tok = len(out.token_ids)
    return {
        "text": out.text,
        "tokens": n_tok,
        "gen_seconds": round(dt, 3),
        "tok_per_s": round(n_tok / dt, 2) if dt > 0 else None,
        "load_seconds": _load_seconds,
        "model": MODEL,
    }


def _int_val(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
