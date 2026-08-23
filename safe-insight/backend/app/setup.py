import json
import psutil
import shutil
import hashlib
import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import httpx

from app import config
from app import network_guard

router = APIRouter(prefix="/setup", tags=["setup"])

def get_catalog():
    catalog_path = config.APP_DIR / "models_catalog.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_settings():
    settings_path = config.DATA_DIR / "settings.json"
    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {}

def save_settings(settings):
    settings_path = config.DATA_DIR / "settings.json"
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

@router.get("/status")
def setup_status():
    settings = get_settings()
    active_filename = settings.get("active_model_filename")
    
    llm_present = False
    if active_filename:
        model_path = config.MODELS_DIR / "llm" / active_filename
        if model_path.exists() and model_path.stat().st_size > 10 * 1024 * 1024:
            llm_present = True
    
    # Fallback: check if any .gguf file exists in the directory
    if not llm_present:
        llm_dir = config.MODELS_DIR / "llm"
        if llm_dir.exists():
            ggufs = [f for f in llm_dir.glob("*.gguf") if f.stat().st_size > 10 * 1024 * 1024]
            if ggufs:
                # Set the first found as active
                settings["active_model_filename"] = ggufs[0].name
                save_settings(settings)
                llm_present = True
                active_filename = ggufs[0].name

    emb_present = False
    if config.EMBEDDING_CACHE_DIR.exists():
        if any(config.EMBEDDING_CACHE_DIR.rglob("config.json")):
            emb_present = True
            
    return {
        "model_present": llm_present and emb_present,
        "active_model": active_filename if llm_present else None
    }

@router.get("/catalog")
def get_catalog_with_ram():
    catalog = get_catalog()
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    for model in catalog:
        model["fits_ram"] = ram_gb >= model["min_ram_gb"]
    return catalog

@router.get("/disk-space")
def disk_space():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(config.MODELS_DIR)
    return {"free_bytes": usage.free, "free_gb": usage.free / (1024 ** 3)}

@router.post("/active-model")
def set_active_model(payload: dict):
    filename = payload.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="Filename required")
    model_path = config.MODELS_DIR / "llm" / filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found on disk")
    
    settings = get_settings()
    settings["active_model_filename"] = filename
    save_settings(settings)
    return {"status": "ok", "active_model": filename}

@router.get("/download/{model_id}")
async def download_model(model_id: str, request: Request):
    catalog = get_catalog()
    model = next((m for m in catalog if m["id"] == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found in catalog")

    async def event_generator():
        target_path = config.MODELS_DIR / "llm" / model["filename"]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        url = f"https://huggingface.co/{model['repo']}/resolve/main/{model['filename']}"
        headers = {}
        with network_guard.sanctioned_download(
            reason="setup model download", 
            allowed_domains={"huggingface.co", ".huggingface.co", ".hf.co"}
        ):
            async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
                try:
                    needs_download = True
                    if target_path.exists():
                        head = await client.head(url)
                        if head.status_code == 200:
                            total_size = int(head.headers.get("Content-Length", 0))
                            if total_size > 0 and target_path.stat().st_size == total_size:
                                needs_download = False
                            else:
                                headers["Range"] = f"bytes={target_path.stat().st_size}-"

                    if needs_download:
                        async with client.stream("GET", url, headers=headers) as response:
                            if response.status_code == 416: # Range Not Satisfiable (already fully downloaded)
                                pass
                            elif response.status_code not in (200, 206):
                                yield f"data: {json.dumps({'error': f'Failed to download: HTTP {response.status_code}'})}\n\n"
                                return
                            else:
                                total_size = int(response.headers.get("Content-Length", 0))
                                downloaded = target_path.stat().st_size if response.status_code == 206 else 0
                                mode = "ab" if response.status_code == 206 else "wb"
                                if mode == "ab":
                                    total_size += downloaded
    
                                with open(target_path, mode) as f:
                                    async for chunk in response.aiter_bytes(chunk_size=1024*1024):
                                        if await request.is_disconnected():
                                            return # Client disconnected
                                        f.write(chunk)
                                        downloaded += len(chunk)
                                        progress = (downloaded / total_size) * 100 if total_size else 0
                                        yield f"data: {json.dumps({'progress': round(progress, 1), 'downloaded': downloaded, 'total': total_size, 'status': 'Downloading...'})}\n\n"
                                        await asyncio.sleep(0) # Yield control

                    yield f"data: {json.dumps({'status': 'Verifying checksum...', 'progress': 100})}\n\n"
                    
                    # Checksum verification in thread so we don't block the event loop entirely
                    def verify():
                        sha256_hash = hashlib.sha256()
                        with open(target_path, "rb") as f:
                            for byte_block in iter(lambda: f.read(1024*1024), b""):
                                sha256_hash.update(byte_block)
                        return sha256_hash.hexdigest()

                    file_hash = await asyncio.to_thread(verify)
                    
                    if file_hash != model["sha256"]:
                        target_path.unlink()
                        yield f"data: {json.dumps({'error': f'Checksum mismatch. Expected {model['sha256']}, got {file_hash}. File deleted.'})}\n\n"
                        return

                    # Set as active model automatically
                    settings = get_settings()
                    settings["active_model_filename"] = model["filename"]
                    save_settings(settings)
                    
                    yield f"data: {json.dumps({'status': 'Downloading embedding model...', 'progress': 100})}\n\n"
                    
                    def download_emb():
                        import huggingface_hub.constants as hc
                        from huggingface_hub import snapshot_download
                        
                        # The huggingface_hub module caches HF_HUB_OFFLINE at import time (when the server booted).
                        # We must actively override the module constant to unstick it for this download thread.
                        hc.HF_HUB_OFFLINE = False
                        
                        snapshot_download(
                            config.EMBEDDING_MODEL_NAME,
                            cache_dir=str(config.EMBEDDING_CACHE_DIR)
                        )
                        
                    try:
                        await asyncio.to_thread(download_emb)
                    except Exception as emb_e:
                        yield f"data: {json.dumps({'error': f'Failed to download embedding model: {str(emb_e)}'})}\n\n"
                        return
                    
                    yield f"data: {json.dumps({'status': 'Complete', 'progress': 100})}\n\n"
                
                except Exception as e:
                    yield f"data: {json.dumps({'error': f'Download failed: {str(e)}'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
