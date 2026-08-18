import base64
import binascii
from contextlib import contextmanager
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import urlparse

def _flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# Opt-in escape hatch: pull the weights from Hugging Face when the RunPod Model
# Cache is not configured on this endpoint. Slow first cold start, but it beats
# a worker that cannot boot. Must be decided before transformers/vllm import.
ALLOW_HF_DOWNLOAD = _flag("ALLOW_HF_DOWNLOAD")
if ALLOW_HF_DOWNLOAD:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
else:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import runpod
from PIL import Image, UnidentifiedImageError
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


SCHEMA_VERSION = "1.1"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# MODEL_NAME overrides both the Model Cache lookup and what we report back.
MODEL_ID = os.environ.get("MODEL_NAME", DEFAULT_MODEL_ID)
CACHE_MODEL_ID = MODEL_ID
CACHE_ROOT = Path(
    os.environ.get(
        "RUNPOD_HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub"
    )
)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "4096"))
# Referentials in the system prompt push a request well past 32k tokens.
MAX_MODEL_LEN = int(os.environ.get("MODEL_SEQ_LEN", "65536"))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "64"))
# Qwen3-VL bills roughly one visual token per 32x32 px after patch merging.
IMAGE_MIN_PIXELS = int(os.environ.get("IMAGE_MIN_PIXELS", 4 * 32 * 32))
IMAGE_MAX_PIXELS = int(os.environ.get("IMAGE_MAX_PIXELS", 400 * 32 * 32))
VIDEO_FPS = 1.0
VIDEO_MIN_PIXELS = 4 * 32 * 32
VIDEO_MAX_PIXELS = 256 * 32 * 32
VIDEO_TOTAL_PIXELS = 20_480 * 32 * 32
VIDEO_SUFFIXES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}

ENGINE = None
PROCESSOR = None

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("qwen3-vl-worker")


# Pilote NVIDIA minimal par version de CUDA embarquee dans l'image.
MIN_DRIVER_FOR_CUDA = {
    "13.0": "580.65.06",
    "12.9": "575.51.03",
    "12.8": "570.26",
    "12.6": "560.28.03",
    "12.4": "550.54.14",
}


def _version_tuple(text):
    parts = []
    for chunk in str(text).split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def log_gpu_environment():
    """Journalise GPU et pilote avant d'initialiser vLLM.

    Sans cela, un pilote hote trop ancien ne se manifeste qu'apres ~30 s de
    chargement, sous la forme d'un CUDA error 803 illisible.
    """
    try:
        import torch
        image_cuda = torch.version.cuda
    except Exception:
        image_cuda = None

    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.warning("nvidia-smi indisponible (%s); pilote hote inconnu.", exc)
        return

    if probe.returncode != 0:
        LOGGER.warning(
            "nvidia-smi a echoue (rc=%s): %s",
            probe.returncode, (probe.stderr or "").strip()[:300],
        )
        return

    rows = [r.strip() for r in probe.stdout.strip().splitlines() if r.strip()]
    driver = None
    for row in rows:
        LOGGER.info("GPU: %s (CUDA de l'image: %s)", row, image_cuda or "inconnue")
        if driver is None and "," in row:
            driver = row.split(",")[-1].strip()

    if not (driver and image_cuda):
        return

    required = MIN_DRIVER_FOR_CUDA.get(image_cuda)
    if not required:
        return

    if _version_tuple(driver) < _version_tuple(required):
        LOGGER.error(
            "PILOTE TROP ANCIEN: l'hote est en %s, or CUDA %s exige >= %s. "
            "Le demarrage de vLLM va echouer avec 'CUDA error 803'. "
            "Corriger cote RunPod: filtre CUDA Version de l'endpoint sur %s et "
            "au-dessus, et/ou changer de type de GPU (les pools recents sont "
            "plus souvent a jour). Aucune image vLLM officielle n'est publiee "
            "pour une version de CUDA plus ancienne.",
            driver, image_cuda, required, image_cuda,
        )
    else:
        LOGGER.info("Pilote %s >= %s requis pour CUDA %s: OK.",
                    driver, required, image_cuda)

    check_cuda_compat(driver)


def check_cuda_compat(driver):
    """Detecte le piege de la *forward compatibility* CUDA.

    /usr/local/cuda*/compat/ contient un libcuda.so prevu pour faire tourner un
    CUDA recent sur un pilote ANCIEN. Si VLLM_ENABLE_CUDA_COMPATIBILITY le fait
    charger alors que le pilote de l'hote est plus RECENT, il masque le libcuda
    injecte par le NVIDIA Container Toolkit et CUDA remonte l'erreur 803, alors
    meme que le pilote est parfaitement valide.
    """
    compat_version = None
    for root in sorted(Path("/usr/local").glob("cuda*/compat")):
        for lib in sorted(root.glob("libcuda.so.*")):
            candidate = lib.name.replace("libcuda.so.", "")
            if candidate and candidate[0].isdigit():
                compat_version = candidate
                LOGGER.info("Bibliotheque CUDA compat presente: %s (%s)",
                            lib, candidate)

    enabled = _flag("VLLM_ENABLE_CUDA_COMPATIBILITY")
    if not enabled:
        if compat_version:
            LOGGER.info(
                "VLLM_ENABLE_CUDA_COMPATIBILITY non active: le libcuda de "
                "l'hote sera utilise. C'est le reglage correct ici."
            )
        return

    if not (driver and compat_version):
        LOGGER.warning(
            "VLLM_ENABLE_CUDA_COMPATIBILITY est actif mais la comparaison des "
            "versions est impossible; en cas d'erreur 803, desactiver ce "
            "drapeau en premier."
        )
        return

    if _version_tuple(driver) >= _version_tuple(compat_version):
        LOGGER.error(
            "CONFLIT CUDA COMPAT: VLLM_ENABLE_CUDA_COMPATIBILITY est actif, la "
            "bibliotheque de compat est en %s, mais le pilote de l'hote est "
            "plus recent (%s). Le libcuda de compat va masquer celui de "
            "l'hote et CUDA echouera en erreur 803, bien que le pilote soit "
            "valide. CORRECTIF: retirer VLLM_ENABLE_CUDA_COMPATIBILITY de "
            "l'image ou de l'endpoint. Ce drapeau ne sert que sur un hote au "
            "pilote TROP ANCIEN.",
            compat_version, driver,
        )
    else:
        LOGGER.info(
            "VLLM_ENABLE_CUDA_COMPATIBILITY actif; compat %s > pilote %s, "
            "usage legitime.", compat_version, driver,
        )


class InputError(ValueError):
    pass


def error_response(code, message, raw_text=None):
    response = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_ID,
        "engine": "vllm",
        "status": "error",
        "error": {"code": code, "message": message},
    }
    if raw_text is not None:
        response["raw_text"] = raw_text
    return response


def describe_cache():
    """What is actually on the volume — the missing half of the old error."""
    lines = []
    volume = Path("/runpod-volume")
    if not volume.is_dir():
        return ["  /runpod-volume n'existe pas: aucun Network Volume n'est "
                "monte sur cet endpoint."]
    if not CACHE_ROOT.is_dir():
        lines.append(f"  {CACHE_ROOT} n'existe pas.")
        try:
            top = sorted(e.name for e in volume.iterdir())[:20]
            lines.append(f"  Contenu de /runpod-volume: {top or '(vide)'}")
        except OSError as exc:
            lines.append(f"  /runpod-volume illisible: {exc}")
        return lines
    try:
        found = sorted(e.name for e in CACHE_ROOT.iterdir() if e.name.startswith("models--"))
    except OSError as exc:
        return [f"  {CACHE_ROOT} illisible: {exc}"]
    lines.append(f"  Modeles presents dans le cache ({len(found)}): "
                 f"{found or '(aucun)'}")
    return lines


def resolve_snapshot_path(model_id):
    # An explicit path wins over everything else.
    override = os.environ.get("MODEL_PATH", "").strip()
    if override:
        if not Path(override).is_dir():
            raise RuntimeError(f"MODEL_PATH={override} n'est pas un dossier.")
        LOGGER.info("MODEL_PATH force le chemin du modele: %s", override)
        return override

    try:
        organisation, name = model_id.split("/", 1)
    except ValueError as exc:
        raise RuntimeError(f"Identifiant Hugging Face invalide: {model_id}") from exc

    model_root = CACHE_ROOT / f"models--{organisation}--{name}"
    snapshots_dir = model_root / "snapshots"
    main_ref = model_root / "refs" / "main"

    if main_ref.is_file():
        candidate = snapshots_dir / main_ref.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            return str(candidate)

    if snapshots_dir.is_dir():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[0])

    if ALLOW_HF_DOWNLOAD:
        LOGGER.warning(
            "%s absent du Model Cache; telechargement depuis Hugging Face "
            "(ALLOW_HF_DOWNLOAD=1). Le premier demarrage a froid sera long.",
            model_id,
        )
        return model_id

    diagnostic = "\n".join(describe_cache())
    raise RuntimeError(
        f"Le Model Cache RunPod ne contient pas {model_id} sous {CACHE_ROOT}.\n"
        f"{diagnostic}\n"
        "  Corrections possibles:\n"
        f"    1. Configurer le Model Cache de l'endpoint sur {model_id} "
        "(RunPod: endpoint > Model (Hugging Face)), puis redeployer.\n"
        "    2. Definir MODEL_NAME sur un modele deja present dans le cache.\n"
        "    3. Definir ALLOW_HF_DOWNLOAD=1 pour telecharger depuis Hugging "
        "Face au demarrage (cold start long, necessite le reseau).\n"
        "    4. Definir MODEL_PATH sur un dossier de poids local."
    )


def load_model_once():
    model_path = resolve_snapshot_path(CACHE_MODEL_ID)
    origin = "Hugging Face" if model_path == CACHE_MODEL_ID else model_path
    LOGGER.info(
        "Chargement de %s depuis %s (max_model_len=%d, max_images=%d, "
        "offline=%s)",
        CACHE_MODEL_ID, origin, MAX_MODEL_LEN, MAX_IMAGES,
        os.environ.get("HF_HUB_OFFLINE"),
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=not ALLOW_HF_DOWNLOAD,
    )
    engine = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
        limit_mm_per_prompt={"image": MAX_IMAGES, "video": 1},
        enable_prefix_caching=True,
        mm_processor_cache_gb=0,
        seed=0,
        enforce_eager=True,
        disable_log_stats=True,
        trust_remote_code=False,
    )
    return engine, processor


def decode_frame(frame, index):
    if not isinstance(frame, dict):
        raise InputError(f"frames[{index}] doit être un objet.")
    if not isinstance(frame.get("id"), str) or not frame["id"].strip():
        raise InputError(f"frames[{index}].id est requis.")
    if not isinstance(frame.get("mime_type"), str) or not frame[
        "mime_type"
    ].startswith("image/"):
        raise InputError(f"frames[{index}].mime_type doit être un type image/*.")
    encoded = frame.get("image_base64")
    if not isinstance(encoded, str) or not encoded:
        raise InputError(f"frames[{index}].image_base64 est requis.")

    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB")
            image.load()
            return image
    except (binascii.Error, UnidentifiedImageError, OSError) as exc:
        raise InputError(f"frames[{index}].image_base64 est invalide.") from exc


def decode_video(payload):
    mime_type = payload.get("mime_type")
    if mime_type not in VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(VIDEO_SUFFIXES))
        raise InputError(f"input.mime_type doit être l'un de: {allowed}.")

    encoded = payload.get("video_base64")
    if not isinstance(encoded, str) or not encoded:
        raise InputError("input.video_base64 est requis.")

    try:
        return base64.b64decode(encoded, validate=True), VIDEO_SUFFIXES[mime_type]
    except binascii.Error as exc:
        raise InputError("input.video_base64 est invalide.") from exc


def validate_response_schema(schema):
    if schema is None:
        return
    if not isinstance(schema, dict) or not schema:
        raise InputError("input.response_schema doit être un JSON Schema objet.")
    if schema.get("type") != "object":
        raise InputError("input.response_schema.type doit valoir object.")
    if not isinstance(schema.get("properties"), dict):
        raise InputError("input.response_schema.properties doit être un objet.")
    try:
        json.dumps(schema, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise InputError("input.response_schema n'est pas sérialisable en JSON.") from exc


def validate_input(job):
    if not isinstance(job, dict) or not isinstance(job.get("input"), dict):
        raise InputError("Le job doit contenir un objet input.")

    payload = job["input"]
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InputError("input.prompt est requis.")

    system_prompt = payload.get("system_prompt")
    if system_prompt is not None and (
        not isinstance(system_prompt, str) or not system_prompt.strip()
    ):
        raise InputError("input.system_prompt doit être une chaîne non vide.")

    validate_response_schema(payload.get("response_schema"))

    for key in ("image_min_pixels", "image_max_pixels"):
        value = payload.get(key)
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise InputError(f"input.{key} doit être un entier positif.")

    frames = payload.get("frames")
    video_url = payload.get("video_url")
    video_base64 = payload.get("video_base64")
    selected_sources = sum(
        (
            isinstance(frames, list) and bool(frames),
            isinstance(video_url, str) and bool(video_url),
            isinstance(video_base64, str) and bool(video_base64),
        )
    )
    if selected_sources != 1:
        raise InputError(
            "Fournir exactement une source: input.frames, input.video_url "
            "ou input.video_base64."
        )

    if isinstance(frames, list) and frames:
        if len(frames) > MAX_IMAGES:
            raise InputError(
                f"input.frames contient {len(frames)} images; "
                f"le worker en accepte {MAX_IMAGES} au maximum."
            )
        images = [decode_frame(frame, index) for index, frame in enumerate(frames)]
        return payload, "frames", images

    if isinstance(video_url, str) and video_url:
        parsed = urlparse(video_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise InputError("input.video_url doit être une URL HTTP(S) absolue.")
        return payload, "video_url", video_url

    video_bytes, suffix = decode_video(payload)
    return payload, "video_base64", (video_bytes, suffix)


def add_optional_system_message(messages, payload):
    system_prompt = payload.get("system_prompt")
    if isinstance(system_prompt, str):
        messages.append({"role": "system", "content": system_prompt.strip()})


def build_frame_messages(payload, images):
    min_pixels = payload.get("image_min_pixels") or IMAGE_MIN_PIXELS
    max_pixels = payload.get("image_max_pixels") or IMAGE_MAX_PIXELS

    content = []
    for frame, image in zip(payload["frames"], images):
        content.append({"type": "text", "text": f"frame_id: {frame['id']}"})
        content.append(
            {
                "type": "image",
                "image": image,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )
    content.append({"type": "text", "text": payload["prompt"].strip()})

    messages = []
    add_optional_system_message(messages, payload)
    messages.append({"role": "user", "content": content})
    return messages


def build_video_messages(payload, video_source):
    content = [
        {
            "type": "video",
            "video": video_source,
            "fps": VIDEO_FPS,
            "min_pixels": VIDEO_MIN_PIXELS,
            "max_pixels": VIDEO_MAX_PIXELS,
            "total_pixels": VIDEO_TOTAL_PIXELS,
        },
        {"type": "text", "text": payload["prompt"].strip()},
    ]
    messages = []
    add_optional_system_message(messages, payload)
    messages.append({"role": "user", "content": content})
    return messages


@contextmanager
def video_reference(source_mode, source):
    if source_mode == "video_url":
        yield source
        return

    video_bytes, suffix = source
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(video_bytes)
            temporary_path = Path(temporary.name)
        yield temporary_path.resolve().as_uri()
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def prepare_vllm_input(messages):
    text = PROCESSOR.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=PROCESSOR.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    multi_modal_data = {}
    if image_inputs is not None:
        multi_modal_data["image"] = image_inputs
    if video_inputs is not None:
        multi_modal_data["video"] = video_inputs
    if not multi_modal_data:
        raise RuntimeError("qwen-vl-utils n'a produit aucun média visuel.")

    processor_kwargs = dict(video_kwargs or {})
    processor_kwargs["do_resize"] = False
    return {
        "prompt": text,
        "multi_modal_data": multi_modal_data,
        "mm_processor_kwargs": processor_kwargs,
    }


def generate_text(messages, response_schema):
    request = prepare_vllm_input(messages)
    sampling_kwargs = {
        "temperature": 0.0,
        "top_k": -1,
        "max_tokens": MAX_NEW_TOKENS,
        "seed": 0,
    }
    if response_schema is not None:
        sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
            json=response_schema
        )

    outputs = ENGINE.generate(
        [request],
        sampling_params=SamplingParams(**sampling_kwargs),
        use_tqdm=False,
    )
    if not outputs or not outputs[0].outputs:
        raise RuntimeError("vLLM n'a produit aucune complétion.")

    output = outputs[0]
    completion = output.outputs[0]
    usage = {
        "prompt_tokens": len(getattr(output, "prompt_token_ids", None) or []),
        "completion_tokens": len(getattr(completion, "token_ids", None) or []),
        "max_model_len": MAX_MODEL_LEN,
    }
    # Present on the vLLM V1 engine; the prefix cache is what makes a static
    # system prompt (referentials) cheap on every call after the first.
    cached = getattr(output, "num_cached_tokens", None)
    if isinstance(cached, int):
        usage["cached_prompt_tokens"] = cached
    return completion.text.strip(), usage


def extract_first_json_object(text):
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_context(raw_text, structured):
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError:
        value = None

    if isinstance(value, dict):
        return value
    if structured:
        return None
    return extract_first_json_object(raw_text)


def handler(job):
    try:
        payload, source_mode, source = validate_input(job)
    except InputError as exc:
        return error_response("INVALID_INPUT", str(exc))

    response_schema = payload.get("response_schema")
    try:
        if source_mode == "frames":
            messages = build_frame_messages(payload, source)
            raw_text, usage = generate_text(messages, response_schema)
        else:
            with video_reference(source_mode, source) as video_source:
                messages = build_video_messages(payload, video_source)
                raw_text, usage = generate_text(messages, response_schema)
    except ValueError as exc:
        message = str(exc)
        if "maximum" in message and "token" in message.lower():
            LOGGER.error("Contexte dépassé: %s", message)
            return error_response(
                "CONTEXT_LENGTH_EXCEEDED",
                "La requête dépasse max_model_len "
                f"({MAX_MODEL_LEN}). Réduire le nombre d'images, "
                "image_max_pixels ou la taille du system_prompt, "
                "ou augmenter MODEL_SEQ_LEN.",
            )
        LOGGER.exception("Échec de l'inférence Qwen3-VL avec vLLM")
        return error_response(
            "INFERENCE_FAILED",
            "L'inférence Qwen3-VL avec vLLM a échoué; voir les logs.",
        )
    except Exception:
        LOGGER.exception("Échec de l'inférence Qwen3-VL avec vLLM")
        return error_response(
            "INFERENCE_FAILED",
            "L'inférence Qwen3-VL avec vLLM a échoué; voir les logs.",
        )

    context = parse_context(raw_text, structured=response_schema is not None)
    if response_schema is not None and context is None:
        return error_response(
            "MODEL_OUTPUT_INVALID_JSON",
            "vLLM n'a pas retourné un objet conforme au schéma demandé.",
            raw_text=raw_text,
        )

    response = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_ID,
        "engine": "vllm",
        "status": "success",
        "source": source_mode,
        "usage": usage,
        "raw_text": raw_text,
    }
    if context is not None:
        response["context"] = context
    return response


if __name__ == "__main__":
    log_gpu_environment()
    ENGINE, PROCESSOR = load_model_once()
    runpod.serverless.start({"handler": handler})
