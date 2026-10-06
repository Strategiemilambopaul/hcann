#!/usr/bin/env python3
"""Telecharge CLIP (605 Mo) avec reprise automatique — avant les demos --use-clip."""

import os
import sys
import time
import urllib.error
import urllib.request

MODEL = "openai/clip-vit-base-patch32"
FILES = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "pytorch_model.bin",
]
BASE = f"https://huggingface.co/{MODEL}/resolve/main/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "clip-vit-base-patch32")
MAX_RETRIES = 8
CHUNK = 512 * 1024  # 512 Ko — plus stable sur connexions lentes


def _format_size(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000} Mo"
    if n >= 1_000:
        return f"{n // 1_000} Ko"
    return f"{n} o"


def _is_complete(name: str, path: str) -> bool:
    if not os.path.isfile(path):
        return False
    size = os.path.getsize(path)
    if name == "pytorch_model.bin":
        return size > 500_000_000
    return size > 0


def download_file(name: str, dest: str) -> None:
    if _is_complete(name, dest):
        print(f"[skip] {name} ({_format_size(os.path.getsize(dest))})")
        return

    url = BASE + name
    tmp = dest + ".part"
    if os.path.isfile(dest) and not os.path.isfile(tmp):
        os.replace(dest, tmp)

    for attempt in range(1, MAX_RETRIES + 1):
        offset = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
        headers = {"User-Agent": "hcann-download/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                if resp.status == 206:
                    total = offset + total
                elif offset and resp.status == 200:
                    # Serveur sans reprise : on repart de zero
                    offset = 0
                    open(tmp, "wb").close()

                mode = "ab" if offset else "wb"
                done = offset
                with open(tmp, mode) as f:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        f.write(block)
                        done += len(block)
                        if total:
                            pct = 100 * done / total
                            print(
                                f"\r[{name}] {pct:5.1f}% ({_format_size(done)}/{_format_size(total)})",
                                end="",
                                flush=True,
                            )
                        else:
                            print(f"\r[{name}] {_format_size(done)}", end="", flush=True)
                print()

            if name == "pytorch_model.bin" and os.path.getsize(tmp) < 500_000_000:
                raise OSError(f"fichier incomplet ({_format_size(os.path.getsize(tmp))})")

            os.replace(tmp, dest)
            print(f"[ok] {dest}")
            return

        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            wait = min(2 ** attempt, 60)
            print(f"\n[retry {attempt}/{MAX_RETRIES}] {name}: {exc} — attente {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"echec apres {MAX_RETRIES} tentatives: {name}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[CLIP] Telechargement vers {OUTPUT_DIR}")
    print("[CLIP] Reprise auto si coupure reseau (WinError 10054)")
    for i, name in enumerate(FILES):
        download_file(name, os.path.join(OUTPUT_DIR, name))
        if i + 1 < len(FILES):
            time.sleep(2)
    print("[CLIP] Termine. Relance: python scripts/demo_nonconfusion.py --use-clip")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[erreur] {exc}", file=sys.stderr)
        print("[info] Relance la meme commande — les fichiers deja telecharges seront ignores/repris.")
        sys.exit(1)
