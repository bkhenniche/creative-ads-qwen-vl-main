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

### Variables d'environnement

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `MODEL_NAME` | `Qwen/Qwen3-VL-8B-Instruct` | Snapshot résolu dans le Model Cache **et** modèle annoncé dans la réponse |
| `MODEL_SEQ_LEN` | `65536` | `max_model_len` de vLLM |
| `MAX_IMAGES` | `64` | Plafond de `input.frames` et `limit_mm_per_prompt` |
| `IMAGE_MAX_PIXELS` | `409600` (400×32×32) | Budget de pixels par frame, ~400 tokens visuels |
| `IMAGE_MIN_PIXELS` | `4096` (4×32×32) | Plancher de pixels par frame |
| `MAX_NEW_TOKENS` | `800` | Longueur maximale de la complétion (surchargeable par `input.max_tokens`) |
| `ALLOW_HF_DOWNLOAD` | *(désactivé)* | `1` pour télécharger les poids depuis Hugging Face si le Model Cache est absent |
| `MODEL_PATH` | *(vide)* | Chemin local de poids, prioritaire sur toute résolution de cache |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Part de la VRAM allouée à vLLM (poids + cache KV) |
| `KV_CACHE_DTYPE` | `auto` | `fp8` divise par deux le coût mémoire du cache KV |

`MODEL_SEQ_LEN` doit rester au-dessus du coût réel d'une requête. Avec des
référentiels dans `system_prompt`, une requête typique tient autour de
**36 000 tokens** (≈31 000 pour le prompt système, ≈324 tokens par frame en
768 px) : les 32 768 d'origine étaient insuffisants.

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
- `image_min_pixels` / `image_max_pixels` : entiers positifs, facultatifs,
  budget de pixels par frame (mode `frames` uniquement) ;
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
  "schema_version": "1.1",
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "engine": "vllm",
  "status": "success",
  "source": "frames",
  "usage": {
    "prompt_tokens": 36121,
    "completion_tokens": 148,
    "cached_prompt_tokens": 30937,
    "max_model_len": 65536
  },
  "raw_text": "{\"brand_name\":\"Marie\"}",
  "context": {"brand_name": "Marie"}
}
```

`usage` sert à deux contrôles. `prompt_tokens` face à `max_model_len` donne la
marge restante avant `CONTEXT_LENGTH_EXCEEDED`. `cached_prompt_tokens` mesure le
préfixe réutilisé par le cache vLLM : sur la deuxième requête d'un worker chaud
partageant le même `system_prompt`, il doit couvrir tout ce prompt système. S'il
reste à zéro, le préfixe n'est pas identique au bit près.

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

## Mode `frames` avec référentiels — usage Pocodex

C'est le mode recommandé pour la labellisation publicitaire : on choisit les
images envoyées au lieu de subir un échantillonnage uniforme, et on met les
référentiels dans `system_prompt` pour qu'ils soient mis en cache.

**1. Choisir les frames.** Le créatif occupe exactement `[0, D)` puis ~1 s de
noir ; `ffmpeg blackdetect` donne `D` sans modèle. Le packshot de fin porte la
marque dans la quasi-totalité des cas, donc on répartit les images sur le corps
du film **et on en force deux dans `[D-3s, D-0.3s]`** :

```python
D = first_black_start(path)                      # 12.0 s
body    = [0.3 + i * (D - 3.3) / 13 for i in range(14)]
endcard = [D - 2.0, D - 0.6]
stamps  = sorted(body + endcard)
```

Extraction en 768 px, désentrelacée — la source est du 1080**i** et le peignage
dégrade la lecture du texte :

```bash
ffmpeg -ss "$T" -i in.mxf -frames:v 1 -vf "yadif,scale=768:-2" -q:v 6 out.jpg
```

**2. Nommer les frames.** `id` est réinjecté tel quel dans le prompt sous la
forme `frame_id: <id>`, juste avant l'image. Des identifiants parlants
(`14_endcard_t10.00`) permettent au prompt de renvoyer vers le packshot sans
autre mécanisme.

**3. Mettre les référentiels dans `system_prompt`.** Le message système est
émis avant les images, donc son préfixe est stable d'un appel à l'autre et vLLM
le réutilise via son cache de préfixe (actif par défaut). Trois conditions :

- préfixe identique **au bit près** — un tri instable, un espace ou un saut de
  ligne en trop suffit à invalider le cache ; générer le bloc une fois comme
  artefact de build ;
- rien de variable avant lui — la durée du spot et l'horodatage vont dans
  `prompt`, jamais dans `system_prompt` ;
- worker chaud — un démarrage à froid paie le préfixe une fois.

**4. Contraindre la sortie.** `response_schema` est transmis à
`StructuredOutputsParams(json=...)`. Les codes doivent être des `enum` du schéma :
un code à 4 lettres est exactement ce qu'un modèle invente de façon plausible, et
`BYDF` contre `BYDD` est invisible en texte libre.

Coût mesuré sur un spot de 12 s : 16 frames en 768 px, corps de requête
**1,08 Mo**, ~36 100 tokens dont ~30 900 de référentiels réutilisables.

## Limites importantes

- Le traitement vidéo suit le flux officiel Qwen3-VL :
  `return_video_kwargs=True`, `return_video_metadata=True`, `do_resize=False`,
  échantillonnage à 1 fps et budgets de pixels bornés.
- Le worker analyse uniquement les images de la vidéo, pas sa piste audio. Une
  transcription audio doit être produite séparément puis fournie dans le prompt.
- RunPod limite le corps de `/run` à 10 Mo et celui de `/runsync` à 20 Mo. Le
  Base64 augmente la taille d'environ un tiers ; utilisez `video_url` pour les
  vidéos plus volumineuses. Un MXF de diffusion brut est hors de portée : 233 Mo
  deviennent 310 Mo en Base64. En mode `frames`, 16 images en 768 px tiennent en
  ~0,9 Mo.
- Une requête trop longue renvoie l'erreur applicative
  `CONTEXT_LENGTH_EXCEEDED` plutôt qu'un échec générique. Réduire le nombre
  d'images, `image_max_pixels` ou le prompt système, ou augmenter
  `MODEL_SEQ_LEN`.
- Les erreurs de validation ou d'inférence gardent le même en-tête de contrat et
  renvoient `status: "error"` avec `error_code` et `error_message`, deux
  **chaines**. La cle `error` de premier niveau est volontairement inutilisee :
  elle est reservee par le SDK runpod (voir Dépannage).

## Dépannage

### `Le Model Cache RunPod ne contient pas ... sous /runpod-volume/huggingface-cache/hub`

Le worker démarre puis sort en `exit code 1` : il ne trouve pas les poids. Le
Model Cache est une propriété **de l'endpoint**, pas de l'image — un endpoint
recréé, ou un second endpoint pointant sur la même image, repart sans cache.

L'erreur liste désormais ce qui est réellement monté :

```
Le Model Cache RunPod ne contient pas Qwen/Qwen3-VL-8B-Instruct sous /runpod-volume/huggingface-cache/hub.
  /runpod-volume n'existe pas: aucun Network Volume n'est monte sur cet endpoint.
  Corrections possibles:
    1. Configurer le Model Cache de l'endpoint ...
```

Deux cas se distinguent immédiatement :

- **`/runpod-volume n'existe pas`** — l'endpoint n'a ni Model Cache ni Network
  Volume. C'est le cas courant sur un endpoint fraîchement créé.
- **`Modeles presents dans le cache: [...]`** — le volume est bien là mais
  contient un autre modèle ; comparer avec `MODEL_NAME`.

**Correctif recommandé.** Sur la page de l'endpoint RunPod, renseigner le modèle
Hugging Face `Qwen/Qwen3-VL-8B-Instruct` dans le Model Cache, puis redéployer.
RunPod pré-télécharge les poids sous
`/runpod-volume/huggingface-cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/` avant
le démarrage des workers, et le chargement reste hors ligne.

**Contournement.** `ALLOW_HF_DOWNLOAD=1` fait télécharger les poids depuis
Hugging Face au démarrage : le worker boote sans Model Cache, au prix d'un
premier démarrage à froid long (~16 Go) répété à chaque worker neuf. Utile pour
débloquer un test, à ne pas garder en production.

`MODEL_PATH` court-circuite toute la résolution et pointe directement un dossier
de poids — utile pour un Network Volume rempli à la main.

### `CUDA error 803: system has unsupported display driver / cuda driver combination`

Le modèle se charge, `EngineCore failed to start`, le worker sort en `exit code 1`.

**Cause reelle, constatee sur cet endpoint : `VLLM_ENABLE_CUDA_COMPATIBILITY=1`.**
Ce n'est *pas* un pilote trop ancien.

Ce drapeau demande a vLLM de charger les bibliotheques de **forward
compatibility** CUDA livrees dans `/usr/local/cuda*/compat/`. Elles existent pour
faire tourner un CUDA recent sur un pilote **ancien**. Le NVIDIA Container
Toolkit, lui, injecte deja dans le conteneur le `libcuda.so` exact de l'hote.
Quand le drapeau est actif, le `libcuda.so` de compat **masque** celui de
l'hote ; si le pilote de l'hote est plus **recent** que la bibliotheque de
compat, le module noyau refuse la combinaison et `cudaGetDeviceCount()` remonte
`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` (803).

Le diagnostic est contre-intuitif : **plus le pilote de l'hote est recent, plus
l'echec est certain**. Resserrer le filtre CUDA de l'endpoint sur 12.9/13.0 —
donc obtenir des pilotes 575/580 — aggrave le probleme au lieu de le resoudre.

Trace observee, ou le pilote est manifestement sain :

```
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 580.126.09 (CUDA de l'image: 12.9)
Pilote 580.126.09 >= 575.51.03 requis pour CUDA 12.9: OK.
...
RuntimeError: ... Error 803: system has unsupported display driver / cuda driver combination
```

**Correctif : retirer `VLLM_ENABLE_CUDA_COMPATIBILITY` de l'image et de
l'endpoint.** Il a ete supprime du `Dockerfile`. Ne l'activer qu'au runtime, au
cas par cas, sur un hote dont le pilote est reellement trop ancien.

Le worker detecte desormais ce piege au demarrage :

```
INFO  Bibliotheque CUDA compat presente: /usr/local/cuda-12.9/compat/libcuda.so.575.51.03
ERROR CONFLIT CUDA COMPAT: ... la bibliotheque de compat est en 575.51.03, mais le
      pilote de l'hote est plus recent (580.126.09). ... CORRECTIF: retirer
      VLLM_ENABLE_CUDA_COMPATIBILITY de l'image ou de l'endpoint.
```

### `MODEL_OUTPUT_TRUNCATED` : le modèle boucle et tronque son JSON

Le modèle repète la même phrase dans un champ texte libre jusqu'à épuiser son
budget de tokens ; le JSON n'est jamais refermé et devient illisible. Observé sur
un `industry_evidence` de ~15 000 caractères, 4 096 tokens consommés en 173 s.

Le décodage contraint garantit la *structure*, pas la *longueur* du contenu
d'une chaîne : sans borne, une boucle reste possible.

**Correctifs, complémentaires :**

- **Côté client, le vrai correctif** : `maxLength` sur chaque champ texte libre
  du `response_schema`. XGrammar l'applique, la boucle devient impossible.
- **Côté worker** : `MAX_NEW_TOKENS` passe de 4096 à **800**, et
  `input.max_tokens` permet de le surcharger par requête.
- L'erreur `MODEL_OUTPUT_TRUNCATED` distingue ce cas de
  `MODEL_OUTPUT_INVALID_JSON` : levée quand `finish_reason == "length"`.
  `usage.finish_reason` est remonté dans toutes les réponses.

### `Failed to return job results. | 400, message='Bad Request'` sur `/job-done`

Le worker demarre, traite le job (`Started.` / `Finished.`), mais le resultat
n'arrive jamais : RunPod refuse le POST vers `/job-done` en **400**, le job est
remis en file et rejoue indefiniment.

**Cause : la cle `error` de premier niveau du dictionnaire retourne.** Le SDK
runpod (>= 1.7) se la reserve et n'accepte qu'une **chaine**. Le handler y
renvoyait un dict `{"code": ..., "message": ...}` ; le POST de resultat devient
alors invalide et echoue en 400.

L'effet est pervers : **toutes les erreurs applicatives devenaient invisibles**.
Le chemin nominal passait, mais la moindre erreur — entree invalide, contexte
depasse, JSON non conforme — disparaissait dans un 400 suivi d'un rejeu, sans
jamais atteindre le client.

**Correctif applique.** `error_response()` n'utilise plus la cle reservee :

```json
{
  "status": "error",
  "error_code": "INVALID_INPUT",
  "error_message": "input.prompt est requis."
}
```

Les erreurs sont aussi journalisees en `ERROR` cote worker, donc visibles dans
les logs RunPod meme si la reponse se perd.

**A retenir pour la suite** : ne jamais placer de structure sous `error` au
premier niveau de la valeur retournee par le handler. Pour les rappels de
limites, `/run` accepte 10 Mo et `/runsync` 20 Mo, en entree comme en sortie.

### `ValueError: ... KV cache is needed, which is larger than the available KV cache memory`

Le moteur demarre, charge les poids, puis refuse d'allouer le cache KV. Ce n'est
plus un probleme de pilote : c'est de l'arithmetique memoire.

Les poids sont incompressibles ; tout le reste du budget VRAM va au cache KV.
Pour `Qwen3-VL-8B` en bf16, comptez **~18 Gio de poids + activations** et
**~144 Kio de cache KV par token**.

| VRAM | `gpu_mem_util` | Cache KV | Capacite | Suffit pour Pocodex (~36 000 tokens) ? |
| --- | --- | --- | --- | --- |
| 24 Gio | 0.90 | 3,6 Gio | ~26 200 tokens | ❌ |
| 24 Gio | 0.95 | 4,8 Gio | ~35 000 tokens | ❌ (de justesse) |
| 24 Gio + **`fp8`** | 0.90 | 3,6 Gio | **~52 400 tokens** | ✅ |
| 32 Gio | 0.90 | 10,8 Gio | ~78 700 tokens | ✅ |
| 48 Gio | 0.90 | 25,2 Gio | ~183 500 tokens | ✅ |

Deux correctifs, au choix :

- **Rester sur 24 Gio** : `KV_CACHE_DTYPE=fp8` et `MODEL_SEQ_LEN=49152`. Le fp8
  double la capacite du cache KV ; l'impact qualite est negligeable pour cette
  tache. Il faut aussi baisser `MODEL_SEQ_LEN` sous la capacite obtenue, sinon
  vLLM refuse toujours de demarrer.
- **Passer sur >= 32 Gio** : les valeurs par defaut fonctionnent telles quelles.

Noter que `GPU_MEMORY_UTILIZATION=0.95` seul **ne suffit pas** sur 24 Gio : il
donne ~35 000 tokens, juste sous le besoin. Ne pas perdre un cycle dessus.

Le worker estime desormais la capacite au demarrage, avant que vLLM n'echoue :

```
INFO  VRAM detectee: 24.0 Gio
INFO  Capacite KV estimee: ~26214 tokens (kv_cache_dtype=auto)
ERROR CACHE KV TROP PETIT: ~26214 tokens disponibles, or une requete Pocodex en
      demande ~36 000. ... (1) KV_CACHE_DTYPE=fp8 (double la capacite, ~52428 tokens)
```

L'estimation est fiable a moins de 0,5 % pres de ce que rapporte vLLM
(26 214 estimes contre 26 304 mesures).

#### Distinguer des deux autres pannes de demarrage

| Symptome | Cause | Correctif |
| --- | --- | --- |
| `Le Model Cache RunPod ne contient pas ...` | L'endpoint n'a pas de Model Cache | Configurer le Model Cache de l'endpoint |
| Erreur **35** `CUDA driver version is insufficient` | Pilote reellement trop ancien | Filtre CUDA Version de l'endpoint |
| Erreur **803** `display driver / cuda driver combination` | Bibliotheques de compat qui masquent le libcuda de l'hote | Retirer `VLLM_ENABLE_CUDA_COMPATIBILITY` |

CUDA distingue les deux derniers cas : un pilote **trop ancien** produit l'erreur
**35** (`cudaErrorInsufficientDriver`), pas la 803. Voir une 803 est donc en
soi un indice qu'il s'agit d'un conflit de bibliotheques, pas d'une question de
version.

Pour reference, les pilotes minimaux par version de CUDA de l'image :

| CUDA de l'image | Pilote minimal |
| --- | --- |
| 13.0 | 580.65.06 |
| 12.9 | 575.51.03 |
| 12.8 | 570.26 |
| 12.6 | 560.28.03 |

Références : [Qwen3-VL — Process Videos](https://github.com/QwenLM/Qwen3-VL#process-videos),
[vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs),
[RunPod Serverless API](https://docs.runpod.io/serverless/endpoints/send-requests).
