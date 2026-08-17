# Qwen3-VL vLLM — worker RunPod Serverless

Worker Queue-based pour analyser une vidéo ou une série d'images avec
[`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
L'inférence utilise vLLM et ses structured outputs JSON Schema.

- Endpoint déployé : `creative-ads-qwen-vl`
- Endpoint ID : `lfc5g5t8u3p9bk`
- Image de base : `vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404`
- Model Cache RunPod : `Qwen/Qwen3-VL-8B-Instruct`
- GPU : A6000 ou A40, 48 Go, un GPU par worker
- Timeout d'exécution : 300 s

Le build ne copie et ne télécharge aucun poids. Au démarrage, le handler
résout exclusivement le snapshot présent dans le Model Cache RunPod sous
`/runpod-volume/huggingface-cache/hub`, puis charge le modèle une seule fois en
BF16.

## Contrat du handler

Le handler ne contient aucune consigne système ni aucun schéma métier. Tout
vient de l'appel :

- `prompt` : chaîne non vide, obligatoire ;
- `system_prompt` : chaîne non vide, facultative ;
- `response_schema` : JSON Schema de type `object`, facultatif ;
- exactement une source parmi `video_base64`, `video_url` et `frames`.

Pour `video_base64`, `mime_type` accepte `video/mp4`, `video/quicktime` ou
`video/webm`. Pour `frames`, chaque entrée contient `id`, `mime_type` et
`image_base64`.

Quand `response_schema` est présent, il est transmis à
`StructuredOutputsParams(json=...)` de vLLM. Un client Python peut donc envoyer
directement `MonModelePydantic.model_json_schema()` ; Pydantic n'a pas besoin
d'être installé dans le worker.

Réponse applicative réussie :

```json
{
  "schema_version": "1.0",
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "engine": "vllm",
  "status": "success",
  "raw_text": "{\"brand_name\":\"Marie\"}",
  "context": {"brand_name": "Marie"}
}
```

Sans schéma, une réponse textuelle reste un succès dans `raw_text`. Si elle
contient un objet JSON, celui-ci est aussi exposé dans `context`. Avec un schéma,
une sortie non parsable produit l'erreur applicative
`MODEL_OUTPUT_INVALID_JSON`.

## Générer `payload.json` avec la vidéo du dossier

Placez la vidéo à côté du README, par exemple sous
`qwen3vl-test-291259134.mp4`, puis exécutez :

```bash
export VIDEO_PATH="./qwen3vl-test-291259134.mp4"

python3 - "$VIDEO_PATH" <<'PY'
import base64
import json
from pathlib import Path
import sys

video_path = Path(sys.argv[1])
schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "brand_name": {"type": ["string", "null"]},
        "summary": {"type": "string"},
        "visible_text": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["brand_name", "summary", "visible_text"],
}
payload = {
    "input": {
        "video_base64": base64.b64encode(video_path.read_bytes()).decode("ascii"),
        "mime_type": "video/mp4",
        "system_prompt": (
            "Analyse uniquement ce qui est visible. N'invente rien et réponds "
            "avec l'objet JSON imposé."
        ),
        "prompt": (
            "Analyse cette publicité en français. Identifie la marque, résume "
            "la narration visuelle et transcris les textes lisibles."
        ),
        "response_schema": schema,
    },
    "policy": {"ttl": 3_600_000},
}
Path("payload.json").write_text(
    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
)
print(f"payload.json créé ({Path('payload.json').stat().st_size} octets)")
PY
```

La conversion Base64 est faite en mémoire par Python, sans passer la longue
chaîne comme argument de commande. Vérifiez ensuite que vous êtes bien dans le
dossier qui contient le fichier :

```bash
test -r payload.json && jq -e '.input.video_base64 | length > 0' payload.json
```

## Appeler l'endpoint avec cURL

Ne stockez jamais la clé API dans le dépôt :

```bash
export RUNPOD_API_KEY="<RUNPOD_API_KEY>"
export ENDPOINT_ID="lfc5g5t8u3p9bk"
```

Santé de l'endpoint :

```bash
curl --silent --show-error --fail-with-body \
  "https://api.runpod.ai/v2/${ENDPOINT_ID}/health" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" |
jq .
```

Soumission asynchrone recommandée :

```bash
test -r payload.json || { echo "payload.json introuvable dans $(pwd)" >&2; exit 1; }

curl --silent --show-error --fail-with-body \
  --request POST \
  --url "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header "Content-Type: application/json" \
  --data-binary @payload.json \
  --output submit-response.json

jq . submit-response.json
export JOB_ID="$(jq -er '.id' submit-response.json)"
echo "JOB_ID=${JOB_ID}"
```

Polling robuste. Contrairement à une boucle qui donne directement la réponse
à `jq`, celle-ci affiche le corps brut et s'arrête si RunPod renvoie une page
HTML, une redirection ou un code HTTP inattendu :

```bash
while true; do
  HTTP_CODE="$(curl --silent --show-error \
    --url "https://api.runpod.ai/v2/${ENDPOINT_ID}/status/${JOB_ID}" \
    --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
    --output job-result.tmp \
    --write-out '%{http_code}')"

  if [ "$HTTP_CODE" != "200" ]; then
    echo "RunPod a renvoyé HTTP ${HTTP_CODE}:" >&2
    sed -n '1,40p' job-result.tmp >&2
    exit 1
  fi
  if ! jq -e . job-result.tmp >/dev/null 2>&1; then
    echo "La réponse RunPod n'est pas du JSON:" >&2
    sed -n '1,40p' job-result.tmp >&2
    exit 1
  fi

  mv job-result.tmp job-result.json
  STATUS="$(jq -r '.status // "UNKNOWN"' job-result.json)"
  jq '{id,status,delayTime,executionTime,workerId,error}' job-result.json

  case "$STATUS" in
    COMPLETED|FAILED|CANCELLED|TIMED_OUT) break ;;
  esac
  sleep 4
done
```

Vérification du contrat final :

```bash
jq '{
  runpod_status: .status,
  schema_version: .output.schema_version,
  model: .output.model,
  engine: .output.engine,
  application_status: .output.status,
  context: .output.context,
  raw_text: .output.raw_text,
  error: (.output.error // .error)
}' job-result.json
```

## Exemple Python avec Pydantic

```python
import base64
import os
import time
from pathlib import Path

import requests
from pydantic import BaseModel, ConfigDict


class Analysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brand_name: str | None
    summary: str
    visible_text: list[str]


endpoint_id = os.environ["ENDPOINT_ID"]
headers = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}
video = base64.b64encode(Path("qwen3vl-test-291259134.mp4").read_bytes()).decode()

payload = {
    "input": {
        "video_base64": video,
        "mime_type": "video/mp4",
        "prompt": "Analyse factuellement cette publicité en français.",
        "response_schema": Analysis.model_json_schema(),
    },
    "policy": {"ttl": 3_600_000},
}
submit = requests.post(
    f"https://api.runpod.ai/v2/{endpoint_id}/run",
    headers=headers,
    json=payload,
    timeout=30,
)
submit.raise_for_status()
job_id = submit.json()["id"]

while True:
    response = requests.get(
        f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    job = response.json()
    if job["status"] in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
        break
    time.sleep(4)

if job["status"] != "COMPLETED" or job["output"]["status"] != "success":
    raise RuntimeError(job)
analysis = Analysis.model_validate(job["output"]["context"])
print(analysis.model_dump_json(indent=2))
```

## Autres sources visuelles

Vidéo accessible par URL HTTPS :

```json
{
  "input": {
    "video_url": "https://example.com/ad.mp4",
    "prompt": "Décris cette publicité en français."
  }
}
```

Frames déjà extraites :

```json
{
  "input": {
    "frames": [
      {
        "id": "frame-001",
        "mime_type": "image/jpeg",
        "image_base64": "<BASE64_JPEG>"
      }
    ],
    "prompt": "Analyse ces frames en français."
  }
}
```

## Limites importantes

- Le traitement vidéo suit le flux officiel Qwen3-VL :
  `return_video_kwargs=True`, `return_video_metadata=True`, `do_resize=False`,
  échantillonnage à 1 fps et budgets de pixels bornés.
- Le worker analyse uniquement les images de la vidéo, pas sa piste audio. Une
  transcription audio doit être produite séparément puis fournie dans le prompt.
- RunPod limite le corps de `/run` à 10 Mo et celui de `/runsync` à 20 Mo. Le
  Base64 augmente la taille d'environ un tiers ; utilisez `video_url` pour les
  vidéos plus volumineuses.
- Les erreurs de validation ou d'inférence gardent le même en-tête de contrat et
  renvoient `status: "error"` avec `error.code` et `error.message`.

Références : [Qwen3-VL — Process Videos](https://github.com/QwenLM/Qwen3-VL#process-videos),
[vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs),
[RunPod Serverless API](https://docs.runpod.io/serverless/endpoints/send-requests).
