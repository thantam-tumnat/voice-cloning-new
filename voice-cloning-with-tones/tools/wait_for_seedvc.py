"""Utility script to poll SeedVC worker health endpoint until it is ready."""
import sys
import time
import urllib.request

def wait_for_seedvc(url: str = "http://127.0.0.1:8022/health", timeout_secs: int = 120) -> int:
    print(f"Waiting for SeedVC worker ({url}) to initialize...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout_secs:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print(f" Ready in {time.time() - start:.1f}s!", flush=True)
                    return 0
        except Exception:
            pass
        time.sleep(2)
        print(".", end="", flush=True)

    print(f"\nWarning: SeedVC worker did not respond within {timeout_secs}s. Proceeding...", flush=True)
    return 1

if __name__ == "__main__":
    target_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8022/health"
    sys.exit(wait_for_seedvc(target_url))
