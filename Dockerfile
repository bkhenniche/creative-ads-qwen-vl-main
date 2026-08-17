FROM vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    FORCE_QWENVL_VIDEO_READER=torchcodec \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
    MODEL_SEQ_LEN=32768 \
    OMP_NUM_THREADS=1 \
    MAX_JOBS=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN uv pip install --system --no-cache -r /app/requirements.txt

COPY handler.py /app/handler.py

ENTRYPOINT ["python3", "-u", "/app/handler.py"]
CMD []
