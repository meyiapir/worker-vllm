# GLM-5.2 serverless worker: minimal RunPod handler over vLLM's STABLE python API.
# Built on vLLM's own 0.23.0 image (engine + flashinfer preinstalled), so there is
# zero "wrapper x engine" version skew — the handler imports only vllm.LLM /
# vllm.SamplingParams, never vllm.entrypoints.* (which is what broke the stock worker).
FROM vllm/vllm-openai:v0.23.0

RUN pip install --no-cache-dir runpod

COPY handler_glm.py /handler_glm.py

# The base image's entrypoint is vLLM's OpenAI api_server; replace it with our handler.
ENTRYPOINT []
CMD ["python3", "/handler_glm.py"]
