"""Comni Model Hub — registry, verification, and HuggingFace download engine.

Independent of PyObjC; usable from both Desktop App and CLI.
"""

import hashlib
import json
import logging
import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("comni.model_hub")

# ── Paths ────────────────────────────────────────────────

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_REGISTRY_PATH = _ASSETS_DIR / "model_registry.json"

_COMNI_HOME = Path.home() / ".comni"
_MODELS_HOME = _COMNI_HOME / "models"
_CACHE_DIR = _COMNI_HOME / "cache"
_COMNI_CONFIG_PATH = _COMNI_HOME / "config.json"

HF_MAIN_ENDPOINT = "https://huggingface.co"
MANIFEST_CACHE_HOURS = 24
HEAD_TIMEOUT_SEC = 5

ProgressCallback = Callable[[str, int, int], None]  # (filename, downloaded, total)


# ── Registry ─────────────────────────────────────────────

def load_registry() -> dict:
    """Load the bundled model_registry.json."""
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def list_available_models() -> List[dict]:
    """Return all model specs from the registry."""
    return load_registry().get("models", [])


def get_model_spec(model_id: str) -> Optional[dict]:
    """Look up a single model by its id."""
    for m in list_available_models():
        if m["id"] == model_id:
            return m
    return None


def match_spec_by_dir(dir_name: str) -> Optional[dict]:
    """Match a local directory name to a registry model spec."""
    for m in list_available_models():
        if m.get("dir_name") == dir_name:
            return m
    name_lower = dir_name.lower()
    for m in list_available_models():
        if m.get("dir_name", "").lower() == name_lower:
            return m
    return None


# ── Comni Config (App-level, separate from server config) ─

def load_comni_config() -> dict:
    if _COMNI_CONFIG_PATH.exists():
        try:
            with open(_COMNI_CONFIG_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_comni_config(cfg: dict):
    _COMNI_HOME.mkdir(parents=True, exist_ok=True)
    with open(_COMNI_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


def get_hf_mirror() -> str:
    return load_comni_config().get("hf_mirror", "")


# ── Verification ─────────────────────────────────────────

@dataclass
class VerifyResult:
    complete: bool = True
    missing: List[str] = field(default_factory=list)
    size_mismatch: List[str] = field(default_factory=list)
    verified: List[str] = field(default_factory=list)
    has_audio: bool = False
    has_tts: bool = False
    has_vision: bool = False
    has_vision_ane: bool = False
    llm: Optional[str] = None


def _check_file(model_path: Path, rel_path: str, expected_size: int) -> str:
    """Return '' if OK, or a description of the issue."""
    fp = model_path / rel_path
    if not fp.exists():
        return "missing"
    if expected_size > 0:
        actual = fp.stat().st_size
        if actual != expected_size:
            return f"size mismatch (expected {expected_size}, got {actual})"
    return ""


def verify_model_from_spec(model_dir: str, spec: dict,
                           chosen_quant: Optional[str] = None) -> VerifyResult:
    """Verify a model directory against a registry spec.

    Quick check: file existence + size comparison. No SHA256.
    """
    result = VerifyResult()
    model_path = Path(model_dir)
    if not model_path.exists():
        result.complete = False
        result.missing.append("(directory not found)")
        return result

    # LLM variant
    llm_found = False
    for v in spec.get("llm_variants", []):
        fp = model_path / v["file"]
        if fp.exists():
            issue = _check_file(model_path, v["file"], v.get("size", 0))
            if not issue:
                result.llm = v["file"]
                result.verified.append(v["file"])
                llm_found = True
                if chosen_quant and v.get("quant") == chosen_quant:
                    break
                if not chosen_quant:
                    break
            else:
                result.size_mismatch.append(v["file"])
    if not llm_found:
        result.complete = False
        result.missing.append("LLM GGUF (no variant found)")

    # Required files
    for entry in spec.get("required_files", []):
        rel = entry["path"]
        issue = _check_file(model_path, rel, entry.get("size", 0))
        if issue == "missing":
            result.complete = False
            result.missing.append(rel)
        elif issue:
            result.size_mismatch.append(rel)
        else:
            result.verified.append(rel)

    # Optional files (check but don't fail)
    for entry in spec.get("optional_files", []):
        rel = entry["path"]
        if entry.get("type") == "directory":
            if (model_path / rel).is_dir():
                result.verified.append(rel)
        else:
            issue = _check_file(model_path, rel, entry.get("size", 0))
            if not issue:
                result.verified.append(rel)

    # Component flags
    components = spec.get("components", {})
    for comp_name, files in components.items():
        all_ok = all((model_path / f).exists() for f in files)
        if comp_name == "audio":
            result.has_audio = all_ok
        elif comp_name == "tts":
            result.has_tts = all_ok
        elif comp_name == "vision":
            result.has_vision = all_ok
        elif comp_name == "vision_ane":
            result.has_vision_ane = all_ok

    return result


def verify_model_generic(model_dir: str) -> VerifyResult:
    """Fallback verification for models not in the registry."""
    result = VerifyResult()
    model_path = Path(model_dir)
    if not model_path.exists():
        result.complete = False
        result.missing.append("(directory not found)")
        return result

    for pattern in ["*Q4_K_M*.gguf", "*Q4_K_S*.gguf", "*Q8_0*.gguf", "*F16*.gguf"]:
        matches = [m for m in model_path.glob(pattern) if m.parent == model_path]
        if matches:
            result.llm = matches[0].name
            result.verified.append(matches[0].name)
            break
    if not result.llm:
        all_gguf = list(model_path.glob("*.gguf"))
        llm_candidates = [f for f in all_gguf
                          if not any(x in f.stem.lower()
                                     for x in ("audio", "vision", "tts", "projector"))]
        if llm_candidates:
            result.llm = llm_candidates[0].name
            result.verified.append(llm_candidates[0].name)
        else:
            result.complete = False
            result.missing.append("LLM GGUF model file")

    # Probe common component patterns
    for d in model_path.iterdir():
        if not d.is_dir():
            continue
        gguf_files = list(d.glob("*.gguf"))
        name_lower = d.name.lower()
        if "audio" in name_lower and gguf_files:
            result.has_audio = True
        elif "tts" in name_lower and gguf_files:
            result.has_tts = True
        elif "vision" in name_lower and gguf_files:
            result.has_vision = True
            mlmodelc = list(d.glob("*.mlmodelc"))
            if mlmodelc and mlmodelc[0].is_dir():
                result.has_vision_ane = True

    return result


def verify_model(model_dir: str, spec: Optional[dict] = None) -> VerifyResult:
    """Unified entry point: uses spec if available, else generic scan."""
    if spec is None:
        dir_name = Path(model_dir).name
        spec = match_spec_by_dir(dir_name)
    if spec:
        return verify_model_from_spec(model_dir, spec)
    return verify_model_generic(model_dir)


def sha256_file(filepath: str, progress_cb: Optional[ProgressCallback] = None) -> str:
    """Compute SHA256 of a file. Optionally report progress."""
    h = hashlib.sha256()
    total = os.path.getsize(filepath)
    done = 0
    name = Path(filepath).name
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if progress_cb:
                progress_cb(name, done, total)
    return h.hexdigest()


# ── HF Manifest (remote file metadata) ──────────────────

def _manifest_cache_path(hf_repo: str) -> Path:
    safe = hf_repo.replace("/", "_")
    return _CACHE_DIR / f"manifest_{safe}.json"


def fetch_remote_manifest(hf_repo: str, force: bool = False) -> Dict[str, dict]:
    """Fetch file metadata from HF API. Returns {rel_path: {size, sha256}}.

    Caches result locally for MANIFEST_CACHE_HOURS.
    """
    cache_file = _manifest_cache_path(hf_repo)
    if not force and cache_file.exists():
        try:
            with open(cache_file, encoding="utf-8") as f:
                cached = json.load(f)
            age_h = (time.time() - cached.get("_ts", 0)) / 3600
            cfg = load_comni_config()
            max_age = cfg.get("manifest_cache_hours", MANIFEST_CACHE_HOURS)
            if age_h < max_age:
                data = dict(cached)
                data.pop("_ts", None)
                return data
        except Exception:
            pass

    logger.info("Fetching manifest from HF: %s", hf_repo)
    manifest = {}
    try:
        from huggingface_hub import HfApi
        api = HfApi(endpoint=HF_MAIN_ENDPOINT)
        info = api.model_info(hf_repo, files_metadata=True)
        for sib in info.siblings or []:
            if not sib.rfilename.endswith(".gguf"):
                continue
            entry = {"size": sib.size or 0}
            if sib.lfs:
                entry["sha256"] = sib.lfs.get("sha256", "") if isinstance(sib.lfs, dict) else getattr(sib.lfs, "sha256", "")
            manifest[sib.rfilename] = entry
    except Exception as e:
        logger.warning("Failed to fetch manifest: %s", e)
        return manifest

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    to_save = dict(manifest)
    to_save["_ts"] = time.time()
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2)
    except Exception:
        pass
    return manifest


# ── Download Engine ──────────────────────────────────────

CHUNK_SIZE = 1024 * 1024          # 1 MB per read
SPEED_TEST_BYTES = 2 * 1024 * 1024  # test with first 2 MB
SPEED_TEST_TIMEOUT = 15           # seconds to download test bytes
MIN_SPEED_BPS = 200 * 1024        # 200 KB/s — below this, switch to mirror
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 60
MAX_PARALLEL = 4                  # concurrent download threads
MAX_RETRIES = 3                   # retry on transient network errors
RETRY_BACKOFF = (2, 5, 10)       # seconds to wait between retries


@dataclass
class DownloadProgress:
    filename: str = ""
    file_index: int = 0
    total_files: int = 0
    bytes_done: int = 0
    bytes_total: int = 0
    speed_bps: float = 0.0
    active_workers: int = 0
    status: str = "pending"   # pending | downloading | verifying | done | error
    error: str = ""


def _resolve_download_url(hf_repo: str, rel_path: str, endpoint: str) -> str:
    return f"{endpoint}/{hf_repo}/resolve/main/{rel_path}"


class ModelDownloader:
    """Parallel model downloader with speed-based mirror fallback and HF HEAD
    requests for download count."""

    def __init__(self, spec: dict, quant: str, dest_dir: Optional[str] = None,
                 mirror_url: str = ""):
        self.spec = spec
        self.quant = quant
        self.dest_dir = Path(dest_dir) if dest_dir else _MODELS_HOME / spec["dir_name"]
        self.mirror_url = mirror_url or get_hf_mirror()
        self.hf_repo = spec["hf_repo"]
        self.progress = DownloadProgress()
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._progress_cb: Optional[Callable[[DownloadProgress], None]] = None
        self._use_mirror: Optional[bool] = None  # None = not yet decided
        self._lock = threading.Lock()
        self._file_progress: Dict[str, int] = {}   # per-file bytes done
        self._completed_count = 0
        self._speed_window_bytes = 0
        self._speed_window_start = 0.0

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self):
        self._cancel.set()

    def _files_to_download(self) -> List[dict]:
        """Build ordered list of files: chosen LLM variant + required + optional.

        Skips entries with type=directory (handled separately by _download_directories).
        Also skips platform-specific entries not matching current OS.
        """
        import platform as _plat
        current_platform = _plat.system().lower()
        files = []
        for v in self.spec.get("llm_variants", []):
            if v["quant"] == self.quant:
                files.append({"path": v["file"], "size": v.get("size", 0)})
                break
        for entry in self.spec.get("required_files", []):
            files.append({"path": entry["path"], "size": entry.get("size", 0)})
        for entry in self.spec.get("optional_files", []):
            if entry.get("type") == "directory":
                continue
            plat = entry.get("platform", "")
            if plat and plat != current_platform:
                continue
            files.append({"path": entry["path"], "size": entry.get("size", 0)})
        return files

    def _directories_to_download(self) -> List[dict]:
        """Return optional directory entries (e.g. .mlmodelc) for current platform."""
        import platform as _plat
        current_platform = _plat.system().lower()
        dirs = []
        for entry in self.spec.get("optional_files", []):
            if entry.get("type") != "directory":
                continue
            plat = entry.get("platform", "")
            if plat and plat != current_platform:
                continue
            dirs.append(entry)
        return dirs

    def _download_directory(self, dir_entry: dict):
        """Download a directory tree (e.g. .mlmodelc) from HF repo."""
        rel_dir = dir_entry["path"]
        dest_dir = self.dest_dir / rel_dir

        if dest_dir.is_dir() and any(dest_dir.iterdir()):
            logger.info("Skipping directory (exists): %s", rel_dir)
            return

        logger.info("Downloading directory: %s", rel_dir)
        try:
            from huggingface_hub import HfApi
            endpoint = self.mirror_url if self._use_mirror else HF_MAIN_ENDPOINT
            api = HfApi(endpoint=endpoint)
            from huggingface_hub import RepoFile
            tree = api.list_repo_tree(self.hf_repo, path_in_repo=rel_dir, recursive=True)
            file_list = []
            for item in tree:
                if isinstance(item, RepoFile):
                    file_list.append({
                        "path": item.rfilename,
                        "size": getattr(item, "size", 0) or 0,
                    })
            if not file_list:
                logger.warning("No files found in HF directory: %s", rel_dir)
                return

            for finfo in file_list:
                if self._cancel.is_set():
                    raise Exception("Cancelled by user")
                self._download_one(finfo)

            logger.info("Directory download complete: %s (%d files)", rel_dir, len(file_list))
        except ImportError:
            logger.warning("huggingface_hub not available, cannot download directory: %s", rel_dir)
        except Exception as e:
            logger.warning("Failed to download directory %s: %s", rel_dir, e)

    def _send_head_for_count(self, rel_path: str):
        """Send HEAD to HF main site in a background thread (non-blocking)."""
        def _head():
            try:
                import urllib.request
                url = _resolve_download_url(self.hf_repo, rel_path, HF_MAIN_ENDPOINT)
                req = urllib.request.Request(url, method="HEAD")
                req.add_header("User-Agent", "Comni/1.0")
                urllib.request.urlopen(req, timeout=10)
                logger.debug("HEAD count OK: %s", rel_path)
            except Exception as e:
                logger.debug("HEAD count failed: %s (%s, non-fatal)", rel_path, e)
        threading.Thread(target=_head, daemon=True).start()

    def _update_aggregate_progress(self, rel_path: str, chunk_bytes: int,
                                   file_done: int):
        """Thread-safe: accumulate per-file progress and push to UI."""
        with self._lock:
            self._file_progress[rel_path] = file_done
            self._speed_window_bytes += chunk_bytes
            total_done = sum(self._file_progress.values())
            now = time.time()
            elapsed = now - self._speed_window_start
            speed = self._speed_window_bytes / max(elapsed, 0.01) if elapsed > 0 else 0
            if elapsed > 5.0:
                self._speed_window_start = now
                self._speed_window_bytes = 0
            self.progress.bytes_done = total_done
            self.progress.speed_bps = speed
            self.progress.file_index = self._completed_count
        self._notify()

    def _stream_download(self, url: str, dest: Path, rel_path: str,
                         expected_size: int, resume_from: int = 0) -> bool:
        """Stream-download a file with real progress reporting.

        Returns True if successful, False if speed too slow (caller should retry
        with mirror). Progress is aggregated across parallel workers.
        """
        import requests

        headers = {"User-Agent": "Comni/1.0"}
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"

        resp = requests.get(url, headers=headers, stream=True,
                            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                            allow_redirects=True)
        resp.raise_for_status()

        done = resume_from
        mode = "ab" if resume_from > 0 else "wb"
        t0 = time.time()
        last_report = t0
        local_bytes = 0

        with open(dest, mode) as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if self._cancel.is_set():
                    resp.close()
                    raise Exception("Cancelled by user")

                f.write(chunk)
                done += len(chunk)
                local_bytes += len(chunk)

                now = time.time()

                # Speed test (only once, for the probe file)
                if self._use_mirror is None and self.mirror_url:
                    if local_bytes >= SPEED_TEST_BYTES:
                        speed = local_bytes / max(now - t0, 0.01)
                        if speed < MIN_SPEED_BPS:
                            resp.close()
                            logger.info("HF too slow (%.0f KB/s), switching to mirror",
                                        speed / 1024)
                            return False
                        logger.info("HF OK (%.0f KB/s), using direct", speed / 1024)
                        self._use_mirror = False
                    elif now - t0 > SPEED_TEST_TIMEOUT:
                        resp.close()
                        logger.info("HF timeout (%ds, %d KB), switching to mirror",
                                    SPEED_TEST_TIMEOUT, local_bytes // 1024)
                        return False

                if now - last_report >= 0.3:
                    self._update_aggregate_progress(rel_path, len(chunk), done)
                    last_report = now

        self._update_aggregate_progress(rel_path, 0, done)
        return True

    def _download_one(self, finfo: dict):
        """Download a single file with retry (called from thread pool)."""
        rel_path = finfo["path"]
        expected_size = finfo.get("size", 0)
        dest_file = self.dest_dir / rel_path

        if dest_file.exists() and expected_size > 0:
            if dest_file.stat().st_size == expected_size:
                logger.info("Skipping (complete): %s", rel_path)
                with self._lock:
                    self._file_progress[rel_path] = expected_size
                    self._completed_count += 1
                    self.progress.file_index = self._completed_count
                self._notify()
                return

        dest_file.parent.mkdir(parents=True, exist_ok=True)
        self._send_head_for_count(rel_path)

        last_err = None
        for attempt in range(MAX_RETRIES):
            if self._cancel.is_set():
                raise Exception("Cancelled by user")

            resume_from = dest_file.stat().st_size if dest_file.exists() else 0
            endpoint = self.mirror_url if self._use_mirror else HF_MAIN_ENDPOINT
            url = _resolve_download_url(self.hf_repo, rel_path, endpoint)

            if attempt == 0:
                logger.info("Downloading %s from %s (resume=%d)",
                            rel_path, endpoint, resume_from)
            else:
                wait = RETRY_BACKOFF[min(attempt - 1, len(RETRY_BACKOFF) - 1)]
                logger.warning("Retry %d/%d for %s (wait %ds, resume=%d)",
                               attempt + 1, MAX_RETRIES, rel_path, wait, resume_from)
                time.sleep(wait)

            try:
                ok = self._stream_download(url, dest_file, rel_path,
                                           expected_size, resume_from)
                if not ok and self.mirror_url:
                    self._use_mirror = True
                    mirror_url = _resolve_download_url(
                        self.hf_repo, rel_path, self.mirror_url)
                    resume_from = dest_file.stat().st_size if dest_file.exists() else 0
                    logger.info("Retrying %s from mirror (resume=%d)",
                                rel_path, resume_from)
                    self._stream_download(mirror_url, dest_file, rel_path,
                                          expected_size, resume_from)

                # Verify size
                if expected_size > 0 and dest_file.exists():
                    actual = dest_file.stat().st_size
                    if actual != expected_size:
                        raise IOError(
                            f"Size mismatch: {rel_path} "
                            f"(expected {expected_size}, got {actual})")

                with self._lock:
                    self._completed_count += 1
                    self.progress.file_index = self._completed_count
                logger.info("Done: %s (%s)", rel_path,
                            _fmt_size(dest_file.stat().st_size))
                return  # success

            except Exception as e:
                last_err = e
                err_msg = str(e)
                if "Cancelled" in err_msg:
                    raise
                logger.warning("Attempt %d failed for %s: %s",
                               attempt + 1, rel_path, err_msg)

        raise Exception(f"Failed after {MAX_RETRIES} retries: {rel_path}: {last_err}")

    def _probe_speed(self, files: List[dict]) -> List[dict]:
        """Download the smallest file first to decide HF vs mirror.

        Returns the remaining files to download in parallel.
        """
        if not self.mirror_url:
            self._use_mirror = False
            return files

        probe = min(files, key=lambda f: f.get("size", 0))
        rest = [f for f in files if f["path"] != probe["path"]]

        logger.info("Speed probe: %s (%s)", probe["path"],
                     _fmt_size(probe.get("size", 0)))
        self._download_one(probe)

        if self._use_mirror is None:
            self._use_mirror = False
        logger.info("Endpoint decided: %s",
                     self.mirror_url if self._use_mirror else HF_MAIN_ENDPOINT)
        return rest

    def _run(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        files = self._files_to_download()
        total_bytes = sum(f.get("size", 0) for f in files)
        self.progress.total_files = len(files)
        self.progress.bytes_total = total_bytes
        self.progress.status = "downloading"
        self._speed_window_start = time.time()
        self._notify()

        try:
            remaining = self._probe_speed(files)
        except Exception as e:
            logger.exception("Speed probe failed")
            self.progress.status = "error"
            self.progress.error = str(e)
            self._notify()
            return

        if self._cancel.is_set():
            self.progress.status = "error"
            self.progress.error = "Cancelled"
            self._notify()
            return

        # Parallel download of remaining files
        if remaining:
            workers = min(MAX_PARALLEL, len(remaining))
            self.progress.active_workers = workers
            self._notify()
            logger.info("Parallel download: %d files, %d workers", len(remaining), workers)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._download_one, f): f for f in remaining}
                for future in as_completed(futures):
                    finfo = futures[future]
                    exc = future.exception()
                    if exc:
                        logger.error("Download failed: %s — %s",
                                     finfo["path"], exc, exc_info=exc)
                        self._cancel.set()
                        self.progress.status = "error"
                        self.progress.error = f"{finfo['path']}: {exc}"
                        self._notify()
                        pool.shutdown(wait=False, cancel_futures=True)
                        return

        self.progress.active_workers = 0

        # Download optional directories (e.g. CoreML .mlmodelc)
        if not self._cancel.is_set():
            for dir_entry in self._directories_to_download():
                if self._cancel.is_set():
                    break
                try:
                    self._download_directory(dir_entry)
                except Exception as e:
                    logger.warning("Optional directory download failed (non-fatal): %s — %s",
                                   dir_entry["path"], e)

        self.progress.status = "verifying"
        self._notify()

        vr = verify_model_from_spec(str(self.dest_dir), self.spec, self.quant)
        if not vr.complete:
            self.progress.status = "error"
            self.progress.error = f"Verification failed: missing {vr.missing}"
            self._notify()
            return

        self.progress.status = "done"
        self.progress.bytes_done = total_bytes
        self._notify()
        logger.info("Model download complete: %s (%s)", self.spec["display_name"], self.quant)

    def _notify(self):
        if self._progress_cb:
            try:
                self._progress_cb(self.progress)
            except Exception:
                pass

    def start(self, progress_cb: Optional[Callable[[DownloadProgress], None]] = None):
        """Start download in background thread."""
        self._progress_cb = progress_cb
        self._cancel.clear()
        self._completed_count = 0
        self._file_progress.clear()
        self._speed_window_bytes = 0
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait(self, timeout: Optional[float] = None):
        if self._thread:
            self._thread.join(timeout=timeout)


def _fmt_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n/1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n/1024**2:.1f} MB"
    return f"{n/1024:.0f} KB"
