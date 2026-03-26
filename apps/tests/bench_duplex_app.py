#!/usr/bin/env python3
"""App 双工 Omni 性能基准 — 用与 C++ 离线测试相同的数据，走 WebSocket 协议"""
import asyncio, json, base64, time, ssl, sys, os
from pathlib import Path
import numpy as np

GATEWAY_WS = os.environ.get("GATEWAY_WS", "wss://localhost:8026")
WORKER_URL = os.environ.get("WORKER_URL", "http://localhost:22420")
APPS_DIR = Path(__file__).resolve().parents[1]
REPO = APPS_DIR.parent
TEST_DIR = REPO / "tools/omni/assets/test_case/omni_test_case"
REF_AUDIO = APPS_DIR / "assets/ref_audio/ref_minicpm_signature.wav"
NUM_CHUNKS = 9

def load_ref_audio_b64():
    import soundfile as sf
    audio, sr = sf.read(str(REF_AUDIO), dtype="float32")
    if sr != 16000:
        ratio = 16000 / sr
        n = int(len(audio) * ratio)
        from scipy.signal import resample
        audio = resample(audio, n).astype(np.float32)
    return base64.b64encode(audio.tobytes()).decode()

def load_wav_b64(path):
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32")
    if sr != 16000:
        ratio = 16000 / sr
        n = int(len(audio) * ratio)
        from scipy.signal import resample
        audio = resample(audio, n).astype(np.float32)
    return base64.b64encode(audio.tobytes()).decode()

def load_jpg_b64(path):
    return base64.b64encode(path.read_bytes()).decode()

async def main():
    import websockets, httpx

    # wait for worker idle
    for _ in range(30):
        try:
            r = httpx.get(f"{WORKER_URL}/health", timeout=5)
            if r.json().get("worker_status") == "idle":
                break
        except:
            pass
        await asyncio.sleep(1)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    sid = f"omni_{int(time.time())}"
    url = f"{GATEWAY_WS}/ws/duplex/{sid}"
    print(f"=== App Duplex Omni Benchmark ===")
    print(f"URL: {url}")
    print(f"Chunks: {NUM_CHUNKS} (audio+video)")
    print()

    t_session = time.time()
    connect_kw = dict(max_size=50_000_000, open_timeout=60, ssl=ctx)
    try:
        connect_kw["proxy"] = None
    except Exception:
        pass
    async with websockets.connect(url, **connect_kw) as ws:
        # queue phase
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            msg = json.loads(raw)
            if msg.get("type") in ("queue_done", "error"):
                break
        if msg.get("type") == "error":
            print(f"Queue error: {msg}")
            return

        t_queue = time.time() - t_session
        print(f"Queue done: {t_queue:.3f}s")

        # prepare
        prepare = {
            "type": "prepare",
            "system_prompt": "You are a helpful assistant.",
            "config": {"max_kv_tokens": 8000},
            "deferred_finalize": True,
        }
        try:
            prepare["tts_ref_audio_base64"] = load_ref_audio_b64()
        except:
            pass

        t0 = time.time()
        await ws.send(json.dumps(prepare))
        raw = await asyncio.wait_for(ws.recv(), timeout=120)
        msg = json.loads(raw)
        t_prepare = time.time() - t0
        print(f"Prepare: {t_prepare:.3f}s -> {msg.get('type')}")
        if msg.get("type") != "prepared":
            print(f"Prepare error: {msg}")
            return

        # send chunks
        chunk_times = []
        for i in range(NUM_CHUNKS):
            idx = f"{i:04d}"
            wav_path = TEST_DIR / f"omni_test_case_{idx}.wav"
            jpg_path = TEST_DIR / f"omni_test_case_{idx}.jpg"

            if not wav_path.exists():
                print(f"  chunk {i}: wav not found, stopping")
                break

            audio_b64 = load_wav_b64(wav_path)
            chunk_msg = {
                "type": "audio_chunk",
                "audio_base64": audio_b64,
                "force_listen": i < 3,
            }
            if jpg_path.exists():
                chunk_msg["frame_base64_list"] = [load_jpg_b64(jpg_path)]

            tc = time.time()
            await ws.send(json.dumps(chunk_msg))
            raw = await asyncio.wait_for(ws.recv(), timeout=120)
            rmsg = json.loads(raw)
            dt = time.time() - tc
            chunk_times.append(dt)

            is_listen = rmsg.get("is_listen", "")
            text = (rmsg.get("text") or "")[:60]
            decision = "listen" if is_listen else f'speak: "{text}"'
            has_frame = "V" if jpg_path.exists() else " "
            print(f"  chunk {i} [{has_frame}]: {dt*1000:7.1f} ms  {decision}")

            await asyncio.sleep(0.05)

        # stop
        await ws.send(json.dumps({"type": "stop"}))
        raw = await asyncio.wait_for(ws.recv(), timeout=30)

    total = time.time() - t_session
    avg = sum(chunk_times) / len(chunk_times) * 1000 if chunk_times else 0
    print()
    print(f"========================================")
    print(f"  App Duplex Benchmark Results")
    print(f"========================================")
    print(f"  Chunks:     {len(chunk_times)} / {NUM_CHUNKS}")
    print(f"  Total:      {total:.3f} s")
    print(f"  Avg/chunk:  {avg:.1f} ms")
    print(f"  Min/chunk:  {min(chunk_times)*1000:.1f} ms")
    print(f"  Max/chunk:  {max(chunk_times)*1000:.1f} ms")
    print(f"========================================")

if __name__ == "__main__":
    asyncio.run(main())
