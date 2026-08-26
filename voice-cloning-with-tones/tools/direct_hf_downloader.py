"""High-speed multi-threaded downloader for HuggingFace model files
using HTTP Range parallel chunk streaming, resume support, and progress reporting.
"""

import concurrent.futures
import os
import sys
import time
from pathlib import Path
import requests

def download_multipart(url: str, dest_path: Path, num_parts: int = 8) -> bool:
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_final = dest_path.with_name(dest_path.name + ".tmp")

    headers_init = {"User-Agent": "Mozilla/5.0"}
    try:
        head = requests.get(url, headers=headers_init, stream=True, timeout=15)
        if head.status_code != 200:
            head.raise_for_status()
        total_size = int(head.headers.get("content-length", 0))
        real_url = head.url
    except Exception as e:
        print(f"  Error resolving {url}: {e}")
        return False

    mb_tot = total_size / (1024 * 1024)
    print(f"  Target Size: {mb_tot:.1f} MB | Spawning {num_parts} parallel connections...")

    part_size = total_size // num_parts
    part_files = [dest_path.parent / f"{dest_path.name}.part{i}" for i in range(num_parts)]

    def download_part(i: int, start: int, end: int) -> int:
        part_file = part_files[i]
        expected_len = end - start + 1
        existing = part_file.stat().st_size if part_file.exists() else 0
        if existing >= expected_len:
            return i

        cur_start = start + existing
        h = {"Range": f"bytes={cur_start}-{end}", "User-Agent": "Mozilla/5.0"}

        for attempt in range(1, 10):
            try:
                r = requests.get(real_url, headers=h, stream=True, timeout=(10, 30))
                if r.status_code not in (200, 206):
                    r.raise_for_status()
                mode = "ab" if existing > 0 else "wb"
                with open(part_file, mode) as pf:
                    for chunk in r.iter_content(chunk_size=128 * 1024):
                        if chunk:
                            pf.write(chunk)
                return i
            except Exception as e:
                time.sleep(2)
                existing = part_file.stat().st_size if part_file.exists() else 0
                cur_start = start + existing
                h["Range"] = f"bytes={cur_start}-{end}"

        raise RuntimeError(f"Part {i} failed after retries")

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parts) as pool:
        futures = []
        for i in range(num_parts):
            s = i * part_size
            e = total_size - 1 if i == num_parts - 1 else (i + 1) * part_size - 1
            futures.append(pool.submit(download_part, i, s, e))

        while True:
            done_count = sum(1 for f in futures if f.done())
            cur_bytes = sum(pf.stat().st_size for pf in part_files if pf.exists())
            dt = time.time() - t0
            speed = (cur_bytes / (1024 * 1024)) / dt if dt > 0 else 0
            pct = (cur_bytes / total_size * 100) if total_size > 0 else 0
            print(f"\r  Parallel Progress: {cur_bytes/(1024*1024):.1f}/{mb_tot:.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s [{done_count}/{num_parts} parts done]", end="", flush=True)
            if done_count == num_parts:
                break
            time.sleep(1)

    print()
    print("  Assembling chunks into final file...", flush=True)
    with open(temp_final, "wb") as outfile:
        for p in part_files:
            with open(p, "rb") as infile:
                while True:
                    data = infile.read(4 * 1024 * 1024)
                    if not data:
                        break
                    outfile.write(data)
            p.unlink(missing_ok=True)

    if dest_path.exists():
        dest_path.unlink()
    temp_final.rename(dest_path)
    print(f"  Saved: {dest_path} ({dest_path.stat().st_size/(1024*1024):.1f} MB in {time.time()-t0:.1f}s)")
    return True

def setup_checkpoint(repo_id: str, filename: str, url: str, base_checkpoints: Path) -> None:
    target_file = base_checkpoints / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "main" / filename
    if target_file.exists() and target_file.stat().st_size > 1000:
        print(f"[CACHE HIT] {repo_id}/{filename} already in checkpoints ({target_file.stat().st_size / (1024*1024):.1f} MB)")
        return

    print(f"\n[DOWNLOADING CHECKPOINT] {repo_id}/{filename} -> {target_file}", flush=True)
    success = download_multipart(url, target_file)
    if not success:
        raise RuntimeError(f"Failed to download {url}")
    print(f"[COMPLETED] {repo_id}/{filename}")

def setup_hf_cache(repo_id: str, filename: str, url: str) -> None:
    cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
    repo_folder = cache_home / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_folder / "snapshots"
    
    ref_file = repo_folder / "refs" / "main"
    if ref_file.exists():
        commit = ref_file.read_text().strip()
    else:
        commit = "main"
        
    target_file = snapshots_dir / commit / filename
    if target_file.exists() and target_file.stat().st_size > 1000:
        print(f"[CACHE HIT] {repo_id}/{filename} already in HF cache ({target_file.stat().st_size / (1024*1024):.1f} MB)")
        return

    print(f"\n[DOWNLOADING HF CACHE] {repo_id}/{filename} -> {target_file}", flush=True)
    success = download_multipart(url, target_file)
    if not success:
        raise RuntimeError(f"Failed to download {url}")
    print(f"[COMPLETED] {repo_id}/{filename}")

def main():
    print("=" * 60)
    print("Fast Multi-Threaded Model Downloader for SeedVC")
    print("=" * 60)

    # 1. RMVPE in checkpoints
    chk_dir = Path("checkpoints").resolve()
    setup_checkpoint(
        "lj1995/VoiceConversionWebUI",
        "rmvpe.pt",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        chk_dir
    )

    # Also copy to seed-vc checkpoints if needed
    seedvc_chk = Path(r"C:\Users\opendream002\Desktop\seed-vc\checkpoints").resolve()
    setup_checkpoint(
        "lj1995/VoiceConversionWebUI",
        "rmvpe.pt",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        seedvc_chk
    )

    # 2. BigVGAN 22k
    setup_hf_cache(
        "nvidia/bigvgan_v2_22khz_80band_256x",
        "bigvgan_generator.pt",
        "https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x/resolve/main/bigvgan_generator.pt"
    )

    # 3. BigVGAN 44k
    setup_hf_cache(
        "nvidia/bigvgan_v2_44khz_128band_512x",
        "bigvgan_generator.pt",
        "https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/resolve/main/bigvgan_generator.pt"
    )

    print("\n" + "=" * 60)
    print("ALL MODELS DOWNLOADED AND READY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
