"""Minimal RunPod serverless handler for GLM-5.2 on vLLM 0.23.x.

Deliberately depends ONLY on vLLM's stable top-level API (`vllm.LLM`,
`vllm.SamplingParams`) — NOT on `vllm.entrypoints.*`, whose module layout
churns between minor releases (that skew is exactly what broke the stock
worker-vllm handler on 0.23). The engine is loaded once at cold start; each
job runs one generation. Enough to smoke-test + measure tok/s vs DeepInfra.

Input (job["input"]):
  {"messages": [{"role":"user","content":"..."}], ...}   # chat, or
  {"prompt": "..."}                                        # raw completion
  + optional: max_tokens, temperature, top_p
Output: {"text": str, "tokens": int, "gen_seconds": float, "tok_per_s": float}
"""
import os
import time

import runpod
from vllm import LLM, SamplingParams


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


MODEL = os.environ.get("MODEL_NAME", "zai-org/GLM-5.2-FP8")

# Loaded once per worker cold start (blocks until the 755GB are pulled + sharded
# across the 8 GPUs). RunPod marks the worker ready only after this returns.
_t0 = time.time()
llm = LLM(
    model=MODEL,
    tensor_parallel_size=_int("TENSOR_PARALLEL_SIZE", 8),
    kv_cache_dtype=os.environ.get("KV_CACHE_DTYPE", "fp8"),
    max_model_len=_int("MAX_MODEL_LEN", 262144),
    gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9")),
    trust_remote_code=True,
)
_load_seconds = round(time.time() - _t0, 1)
print(f"[handler] model loaded in {_load_seconds}s: {MODEL}", flush=True)


def handler(job):
    inp = job.get("input", {}) or {}
    sp = SamplingParams(
        temperature=float(inp.get("temperature", 0.7)),
        top_p=float(inp.get("top_p", 0.95)),
        max_tokens=_int_val(inp.get("max_tokens"), 512),
    )
    t0 = time.time()
    if inp.get("messages"):
        outs = llm.chat(inp["messages"], sp)
    else:
        outs = llm.generate(inp.get("prompt", "Hello"), sp)
    dt = time.time() - t0

    out = outs[0].outputs[0]
    n_tok = len(out.token_ids)
    return {
        "text": out.text,
        "tokens": n_tok,
        "gen_seconds": round(dt, 3),
        "tok_per_s": round(n_tok / dt, 2) if dt > 0 else None,
        "model": MODEL,
        "load_seconds": _load_seconds,
    }


def _int_val(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


runpod.serverless.start({"handler": handler})
