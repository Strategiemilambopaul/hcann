#!/usr/bin/env python3
"""Banc fondateur : l'épisode comme unité de mesure (lexical / merge / RAG / HCANN).

Usage :
  python scripts/demo_nonconfusion.py --use-clip
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.episodic_bench import (
    HcannStore,
    LexicalStore,
    MergeStore,
    RagStore,
    compare_summaries,
    evaluate_pair,
    ingest,
)
from integrations.episodic_memory import EpisodicMemory
from integrations.episodic_scenes import degraded_of, distractors, twin_pairs
from utils.clip_checkpoint import configure_stdout

OUT_PATH = REPO_ROOT / "results" / "episodic" / "nonconfusion.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unité de mesure épisodique — lexical / merge / RAG / HCANN")
    p.add_argument("--use-clip", action="store_true", help="CLIP réel (recommandé)")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--tie-margin", type=float, default=0.05)
    p.add_argument("--dg-mix", type=float, default=0.4)
    p.add_argument("--twin-min-sim", type=float, default=0.85)
    p.add_argument("--tiebreak", choices=("auto", "always", "never"), default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=str(OUT_PATH))
    return p.parse_args()


def _note(summary: dict, encoder: str) -> str:
    rag_conf = summary["rag_confusion_at_1"]
    hcann_conf = summary["hcann_confusion_at_1"]
    rag_hit = summary["rag_hit_at_1"]
    hcann_hit = summary["hcann_hit_at_1"]
    n_ca3 = summary.get("hcann_ca3", 0)
    note = (
        "Mètre : épisode vécu. Quatre barres — lexical, merge (fusion Jaccard), RAG cosine, HCANN — même journal."
        if encoder == "clip"
        else "Encodeur mock : le protocole tourne, ce n'est pas une preuve visuelle. Relancer avec --use-clip."
    )
    if encoder != "clip":
        return note
    if rag_conf == 0:
        note += " RAG n'a pas confondu (CLIP a déjà séparé)."
    if rag_conf > hcann_conf:
        note += " HCANN réduit la confusion exclusive vs RAG."
    elif rag_conf == hcann_conf:
        note += " Confusion exclusive HCANN = RAG."
    if n_ca3:
        note += f" Complétion CA3 sur {n_ca3} requête(s)."
    if hcann_hit >= rag_hit:
        note += " Hit HCANN >= RAG."
    else:
        note += " Hit encore inférieur au RAG."
    note += " Lexical ignore l'image. Merge fusionne les légendes trop proches (geste Mem0)."
    return note


def _print_system_line(name: str, block: dict, indent: str = "  ") -> None:
    print(
        f"{indent}{name:<10} conf {100 * block['confusion_at_1']:5.1f}%  "
        f"hit {100 * block['hit']:5.1f}%  paire@2 {100 * block['pair_at_2']:5.1f}%"
    )


def main() -> int:
    configure_stdout()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    memory = EpisodicMemory.create(
        use_clip=args.use_clip,
        enable_dreaming=False,
        tie_margin=args.tie_margin,
        dg_mix=args.dg_mix,
        twin_min_sim=args.twin_min_sim,
    )
    encoder = "mock" if memory.uses_mock_encoder else "clip"
    if args.use_clip and encoder == "mock":
        print("[nonconfusion] --use-clip demandé mais CLIP indisponible, fallback mock.")

    pairs = twin_pairs()
    extras = distractors()
    hcann = HcannStore(memory, tiebreak=args.tiebreak)
    rag = RagStore(memory)
    lexical = LexicalStore()
    merged = MergeStore()
    n_ep = ingest(hcann, pairs, extras)
    ingest(lexical, pairs, extras)
    ingest(merged, pairs, extras)
    stores = {"rag": rag, "lexical": lexical, "merge": merged, "hcann": hcann}
    rows: list[dict] = []
    for pair in pairs:
        rows.extend(evaluate_pair(stores, pair, k=args.k, degraded_fn=degraded_of))

    summary = compare_summaries(rows)
    summary["note"] = _note(summary, encoder)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "unit": "episodic_nonconfusion",
        "encoder": encoder,
        "k": args.k,
        "tie_margin": args.tie_margin,
        "dg_mix": args.dg_mix,
        "twin_min_sim": args.twin_min_sim,
        "tiebreak": args.tiebreak,
        "seed": args.seed,
        "n_episodes": n_ep,
        "summary": summary,
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 64)
    print("  Unité de mesure — épisodes vécus (lexical / merge / RAG / HCANN)")
    print("=" * 64)
    print(f"  Encodeur     : {encoder}")
    print(f"  Épisodes     : {n_ep}")
    print(f"  Requêtes     : {summary['n_queries']}")
    order = [n for n in ("rag", "lexical", "merge", "hcann") if n in summary.get("systems", {})]
    print("  Global")
    for name in order:
        _print_system_line(name, summary["systems"][name], indent="    ")
    print(
        f"  Chemins HCANN : identity/CA3/ambigu {summary['dg_tiebreak_used']}/{summary['n_queries']}, "
        f"ambigu {summary['hcann_ambiguous']}, ca3 {summary.get('hcann_ca3', 0)}"
    )
    for qname in summary.get("by_query", {}):
        print(f"  [{qname}]")
        for name in order:
            qsys = summary["systems"][name]["by_query"][qname]
            _print_system_line(name, qsys, indent="    ")
    print(f"  {summary['note']}")
    print(f"  Export : {out}")
    vet = pairs[1]["a"]
    probe = memory.deja_vu(text=vet["text"], image=vet["image"])
    print(
        f"  Déjà-vu (épisode vet encodé) : deja_vu={probe['deja_vu']} "
        f"nearest={probe['nearest_id']} cos={probe['nearest_cosine']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
