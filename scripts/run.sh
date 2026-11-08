#!/usr/bin/env bash
# POSIX / macOS / Linux launcher for Curio (equivalent to scripts/run.ps1).
# Never initiates network downloads or model training; uses local cache & frozen checks.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENCODER="siglip"
INDEX=""
DATA="data/snapshots/collection_v2.jsonl"
PROBES=""
PROBE_WEIGHT="-1"
PORT="8000"
CHECK_ONLY="0"
COMPOSITIONAL_PROBES="0"
COMP_LORA="0"
EXPLICIT_ENCODER="0"
EXPLICIT_INDEX="0"
EXPLICIT_PROBES="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --encoder) ENCODER="$2"; EXPLICIT_ENCODER="1"; shift 2 ;;
    --index) INDEX="$2"; EXPLICIT_INDEX="1"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --probes) PROBES="$2"; EXPLICIT_PROBES="1"; shift 2 ;;
    --probe-weight) PROBE_WEIGHT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --compositional) COMPOSITIONAL_PROBES="1"; shift ;;
    --comp-lora) COMP_LORA="1"; shift ;;
    --check-only) CHECK_ONLY="1"; shift ;;
    -h|--help)
      echo "Usage: scripts/run.sh [--compositional] [--comp-lora] [--port 8000] [--check-only] [--encoder siglip|clip] [--index DIR] [--data FILE] [--probes DIR] [--probe-weight W]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${HF_HUB_CACHE:-}" && -d "$REPO_ROOT/models/huggingface/hub" ]]; then
  export HF_HUB_CACHE="$REPO_ROOT/models/huggingface/hub"
fi

PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing .venv/bin/python. Run: uv sync --extra dev" >&2
  exit 1
fi

if [[ ! -f "$DATA" ]]; then
  echo "Missing catalog $DATA. Run python3 scripts/download_resources.py first." >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/web/dist/index.html" ]]; then
  echo "Missing frontend build web/dist/index.html. Run: npm --prefix web run build" >&2
  exit 1
fi

if [[ "$COMP_LORA" == "1" ]]; then
  if [[ "$EXPLICIT_INDEX" == "0" ]]; then
    INDEX="artifacts/siglip_comp_lora"
    EXPLICIT_INDEX="1"
  fi
  if [[ "$EXPLICIT_PROBES" == "0" ]]; then
    PROBES="artifacts/probes/vision_comp_lora"
    EXPLICIT_PROBES="1"
  fi
  if [[ "$PROBE_WEIGHT" == "-1" ]]; then
    PROBE_WEIGHT="0.15"
  fi
  if [[ -z "${CURIO_EXPANDED_INDEX:-}" && -f "$REPO_ROOT/artifacts/siglip_10k_comp_lora/manifest.json" ]]; then
    export CURIO_EXPANDED_INDEX="artifacts/siglip_10k_comp_lora"
  fi
elif [[ "$COMPOSITIONAL_PROBES" == "1" && "$EXPLICIT_PROBES" == "0" ]]; then
  PROBES="artifacts/probes/vision_metadata_adapted"
  EXPLICIT_PROBES="1"
  if [[ "$PROBE_WEIGHT" == "-1" ]]; then
    PROBE_WEIGHT="0.0"
  fi
fi

# Validate index, frozen selection SHA256s, and probes via Python
eval "$("$PYTHON_BIN" - "$ENCODER" "$INDEX" "$DATA" "$PROBES" "$PROBE_WEIGHT" "$EXPLICIT_ENCODER" "$EXPLICIT_INDEX" "$EXPLICIT_PROBES" <<'PY'
import hashlib, json, shlex, sys
from pathlib import Path

encoder, index, data, probes, pw_str, exp_enc, exp_idx, exp_prb = sys.argv[1:9]
pw = float(pw_str)
data_path = Path(data)
data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()

use_vis_sel = False
vis_sel = None
vis_choice = None
sel_path = Path("artifacts/visual_selection.json")
if exp_enc == "0" and exp_idx == "0" and sel_path.is_file():
    vis_sel = json.loads(sel_path.read_text(encoding="utf-8"))
    if vis_sel.get("data_sha256") == data_sha:
        vis_choice = vis_sel["selected"]
        if vis_choice.get("variant") != "image_only":
            raise SystemExit("Frozen production choice must use image_only.")
        model = vis_choice["model"]
        encoder = "clip" if model == "clip" else "siglip"
        index = f"artifacts/{model}"
        if exp_prb == "0":
            probes = str(vis_choice.get("probes") or "")
        if pw < 0:
            pw = float(vis_choice.get("weight", 0.0))
        use_vis_sel = True

if not index:
    index = f"artifacts/{encoder}"

idx_dir = Path(index)
for req in ("image_embeddings.npy", "item_ids.json", "manifest.json"):
    if not (idx_dir / req).is_file():
        raise SystemExit(f"Missing index file: {idx_dir / req}")

manifest = json.loads((idx_dir / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("encoder") != encoder:
    raise SystemExit(f"Index encoder {manifest.get('encoder')} != {encoder}")
if manifest.get("data_sha256") != data_sha:
    raise SystemExit("Catalog/index SHA256 mismatch.")

if use_vis_sel:
    hashes = {k.replace("\\", "/"): v.lower() for k, v in (vis_sel.get("model_files_sha256") or {}).items()}
    def check_file(rel_str: str) -> None:
        rel_norm = rel_str.replace("\\", "/")
        if rel_norm not in hashes:
            raise SystemExit(f"Frozen visual artifact lacks fingerprint: {rel_norm}")
        p = Path(rel_norm)
        if not p.is_file():
            raise SystemExit(f"Frozen visual artifact missing: {rel_norm}")
        actual = hashlib.sha256(p.read_bytes()).hexdigest().lower()
        if actual != hashes[rel_norm]:
            raise SystemExit(f"Frozen visual artifact modified: {rel_norm}")
    for req in ("image_embeddings.npy", "item_ids.json", "manifest.json"):
        check_file(f"{index}/{req}")
    if manifest.get("adapted"):
        ap = str(manifest.get("adapter_path") or "").replace("\\", "/")
        for req in ("adapter_config.json", "adapter_model.safetensors"):
            check_file(f"{ap}/{req}")

if probes:
    rep_path = Path(probes) / "report.json"
    if not rep_path.is_file():
        raise SystemExit(f"Missing probe report: {rep_path}")
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    if rep.get("data_sha256") != data_sha or rep.get("index_sha256") != manifest.get("index_sha256"):
        raise SystemExit("Probe/data/index SHA256 mismatch.")

if pw < 0:
    pw = 0.0

using_frozen_prod = use_vis_sel and exp_prb == "0" and pw == float(vis_choice.get("weight", 0.0))
metrics_path = "artifacts/evaluation/image_audit/selected_test/metrics.json" if using_frozen_prod else ""

for k, v in (
    ("ENCODER", encoder),
    ("INDEX", index),
    ("PROBES", probes),
    ("PROBE_WEIGHT", str(pw)),
    ("METRICS_PATH", metrics_path),
):
    print(f"{k}={shlex.quote(v)}")
PY
)"

export CURIO_ENCODER="$ENCODER"
export CURIO_INDEX="$INDEX"
export CURIO_DATA="$DATA"
export CURIO_PROBES="$PROBES"
export CURIO_PROBE_WEIGHT="$PROBE_WEIGHT"
export CURIO_HOST="127.0.0.1"
export CURIO_PORT="$PORT"
if [[ -n "$METRICS_PATH" ]]; then
  export CURIO_METRICS="$METRICS_PATH"
else
  unset CURIO_METRICS || true
fi
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
if [[ -z "${CURIO_DEVICE:-}" && "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  export CURIO_DEVICE="mps"
fi

echo "Curio: encoder=$CURIO_ENCODER index=$CURIO_INDEX data=$CURIO_DATA probes=$CURIO_PROBES probeWeight=$CURIO_PROBE_WEIGHT"
if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "Resource checks passed."
  exit 0
fi

echo "Open http://127.0.0.1:$PORT . Press Ctrl+C to stop."
exec "$PYTHON_BIN" -m curio.serve
