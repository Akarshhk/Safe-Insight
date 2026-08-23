"""
One-time model downloader.

This is the ONLY script in the project that touches the network, and it is not
part of the runtime path: run it once during setup, then the app never reaches
out again. It deliberately does **not** import :mod:`app.network_guard` - the
guard would block it, which is exactly the behaviour we want everywhere else.

What it fetches
---------------
1. The generation model: a Q4_K_M GGUF quantisation of Phi-3.5-mini-instruct
   (~2.2 GB), into ``backend/models/llm/``.
2. The embedding model: ``BAAI/bge-small-en-v1.5`` (~130 MB), into
   ``backend/models/embeddings/``.

Usage
-----
    python download_model.py                  # both models, defaults
    python download_model.py --llm llama3     # Llama-3.2-3B-Instruct instead
    python download_model.py --embedding-only # skip the 2 GB download
    python download_model.py --llm-only
    python download_model.py --check          # report what is already present

After it finishes you can disconnect from the network permanently.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Keep hub calls online for this script only. app/config.py sets these to "1";
# we must clear them *before* importing anything from the app package.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("SAFE_INSIGHT_MODELS_DIR") or (BACKEND_DIR / "models"))
LLM_DIR = MODELS_DIR / "llm"
EMBEDDING_DIR = MODELS_DIR / "embeddings"

#: GGUF options. Add an entry here to make another model one flag away.
#: All are instruct-tuned, Q4_K_M-quantised, and run on 8 GB of RAM, CPU-only.
LLM_CHOICES: Dict[str, Dict[str, str]] = {
    "phi3": {
        "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "approx_size": "2.2 GB",
        "note": "Default. 3.8B params, strong instruction-following for its size.",
    },
    "llama3": {
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "approx_size": "2.0 GB",
        "note": "Alternative. Slightly smaller and faster; set SAFE_INSIGHT_LLM_FILE to use it.",
    },
}

EMBEDDING_CHOICES: Dict[str, Dict[str, str]] = {
    "bge": {
        "repo_id": "BAAI/bge-small-en-v1.5",
        "approx_size": "130 MB",
        "note": "Default. 384-dim, best retrieval quality per megabyte.",
    },
    "minilm": {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "approx_size": "90 MB",
        "note": "Fallback. Also 384-dim, so the FAISS index stays compatible.",
    },
}


def _fail(message: str) -> None:
    """Print an error and exit non-zero (so a setup script can detect failure)."""
    print(f"\n[ERROR] {message}", file=sys.stderr)
    sys.exit(1)


def download_llm(choice: str) -> Path:
    """
    Download one GGUF file into ``backend/models/llm/``.

    ``hf_hub_download`` resumes partial downloads, so an interrupted 2 GB pull
    can simply be re-run.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        _fail("huggingface-hub is not installed. Run: pip install -r requirements.txt")

    spec = LLM_CHOICES[choice]
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    destination = LLM_DIR / spec["filename"]

    if destination.exists():
        size_gb = destination.stat().st_size / 1_073_741_824
        print(f"[skip] {spec['filename']} already present ({size_gb:.2f} GB)")
        return destination

    print(f"[llm ] Downloading {spec['filename']} (~{spec['approx_size']})")
    print(f"       from {spec['repo_id']}")
    print("       This is the big one. Resume-safe: re-run if it drops.\n")

    path = hf_hub_download(
        repo_id=spec["repo_id"],
        filename=spec["filename"],
        local_dir=str(LLM_DIR),
        # Copy out of the blob cache so the .gguf sits at a plain, stable path
        # that SAFE_INSIGHT_LLM_PATH can point at.
        local_dir_use_symlinks=False,
    )
    print(f"[ok  ] Saved to {path}")
    return Path(path)


def download_embedding(choice: str) -> Path:
    """
    Download a sentence-transformers model into ``backend/models/embeddings/``.

    We load it through ``SentenceTransformer`` rather than pulling raw files so
    the cache layout is exactly what the runtime expects with
    ``local_files_only=True``.
    """
    spec = EMBEDDING_CHOICES[choice]
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(EMBEDDING_DIR)
    os.environ["HF_HOME"] = str(EMBEDDING_DIR)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _fail("sentence-transformers is not installed. Run: pip install -r requirements.txt")

    print(f"[emb ] Downloading {spec['repo_id']} (~{spec['approx_size']})")
    model = SentenceTransformer(spec["repo_id"], cache_folder=str(EMBEDDING_DIR), device="cpu")

    # Prove it works offline-style before we claim success.
    vector = model.encode(["safe insight offline check"], normalize_embeddings=True)
    print(f"[ok  ] Embedding model ready, dimension = {vector.shape[1]}")
    return EMBEDDING_DIR


def report_status() -> None:
    """Print what is already downloaded - the ``--check`` mode."""
    print("Model status")
    print("------------")
    print(f"Models directory: {MODELS_DIR}\n")

    print("Generation models (GGUF):")
    found_llm = False
    for key, spec in LLM_CHOICES.items():
        path = LLM_DIR / spec["filename"]
        if path.exists():
            found_llm = True
            size_gb = path.stat().st_size / 1_073_741_824
            print(f"  [x] {key:8s} {spec['filename']} ({size_gb:.2f} GB)")
        else:
            print(f"  [ ] {key:8s} {spec['filename']} - not downloaded")
    if not found_llm:
        print("      -> Run 'python download_model.py --llm-only' for generated answers.")
        print("         Without it the app still ingests, retrieves and cites (stub mode).")

    print("\nEmbedding models:")
    if EMBEDDING_DIR.exists() and any(EMBEDDING_DIR.rglob("config.json")):
        print(f"  [x] cache populated at {EMBEDDING_DIR}")
    else:
        print(f"  [ ] no embedding model cached at {EMBEDDING_DIR}")
        print("      -> Run 'python download_model.py --embedding-only'. REQUIRED:")
        print("         without it nothing can be indexed or searched.")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Safe Insight's local models. Run once, then stay offline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--llm",
        choices=sorted(LLM_CHOICES),
        default="phi3",
        help="Which GGUF generation model to fetch (default: phi3).",
    )
    parser.add_argument(
        "--embedding",
        choices=sorted(EMBEDDING_CHOICES),
        default="bge",
        help="Which embedding model to fetch (default: bge).",
    )
    parser.add_argument("--llm-only", action="store_true", help="Skip the embedding model.")
    parser.add_argument(
        "--embedding-only",
        action="store_true",
        help="Skip the 2 GB GGUF download (app runs in retrieval-only stub mode).",
    )
    parser.add_argument(
        "--check", action="store_true", help="Report what is already downloaded and exit."
    )
    args = parser.parse_args(argv)

    if args.check:
        report_status()
        return 0

    print("Safe Insight - one-time model download")
    print("======================================")
    print("This is the only step that uses the network. After it completes you")
    print("can disconnect permanently; the app never contacts anything again.\n")

    if not args.llm_only:
        download_embedding(args.embedding)
        print()
    if not args.embedding_only:
        download_llm(args.llm)
        print()

    print("Done. Next steps:")
    print("  1. python -m app.main            # start the backend")
    print("  2. cd ../frontend && npm run tauri dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
