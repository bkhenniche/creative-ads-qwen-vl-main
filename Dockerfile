FROM vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404

# NE PAS activer VLLM_ENABLE_CUDA_COMPATIBILITY ici.
# Ce drapeau charge les bibliotheques de *forward compatibility* CUDA de
# /usr/local/cuda/compat/, concues pour faire tourner un CUDA recent sur un
# pilote ANCIEN. Quand le pilote de l'hote est plus RECENT que ces
# bibliotheques (ex. 580.126.09 face a une image cu129), le libcuda.so de
# compat masque celui injecte par le NVIDIA Container Toolkit, le module noyau
# refuse la combinaison, et CUDA remonte l'erreur 803
# (CUDA_ERROR_SYSTEM_DRIVER_MISMATCH) des cudaGetDeviceCount().
# A n'activer qu'au cas par cas, au runtime, si un hote a un pilote trop ancien.
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    FORCE_QWENVL_VIDEO_READER=torchcodec \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    MODEL_SEQ_LEN=65536 \
    OMP_NUM_THREADS=1 \
    MAX_JOBS=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN uv pip install --system --no-cache -r /app/requirements.txt

COPY handler.py /app/handler.py

ENTRYPOINT ["python3", "-u", "/app/handler.py"]
CMD []
