#!/usr/bin/env python3
"""Complétion de motif : amorce = motif DG stocké, pas un crop CLIP.

Un crop / une légende partielle re-encodés par CLIP ne sont pas un fragment
du motif Hopfield. Ce script sépare le diagnostic CLIP du test CA3 réel.

Usage :
  python scripts/demo_completion.py --use-clip
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
    Ca3Store,
    DgKnnStore,
    HcannStore,
    LexicalStore,
    RagStore,
    compare_summaries,
    evaluate_pattern_cues,
    evaluate_visits,
    ingest_visits,
)
from integrations.episodic_memory import EpisodicMemory
from integrations.episodic_scenes import completion_queries, distinct_visits
from utils.clip_checkpoint import configure_stdout

OUT_PATH = REPO_ROOT / "results" / "episodic" / "completion.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Complétion de motif — DG stocké vs cue CLIP")
    p.add_argument("--use-clip", action="store_true")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=str(OUT_PATH))
    return p.parse_args()


def _print_system_line(name: str, block: dict, indent: str = "  ") -> None:
    print(
        f"{indent}{name:<10} err {100 * block['confusion_at_1']:5.1f}%  "
        f"hit {100 * block['hit']:5.1f}%  hit@2 {100 * block['pair_at_2']:5.1f}%"
    )


def _print_block(title: str, summary: dict, order: tuple[str, ...]) -> None:
    print(title)
    names = [n for n in order if n in summary.get("systems", {})]
    print("  Global  (err = vainqueur unique faux)")
    for name in names:
        _print_system_line(name, summary["systems"][name], indent="    ")
    for qname in summary.get("by_query", {}):
        print(f"  [{qname}]")
        for name in names:
            qsys = summary["systems"][name]["by_query"][qname]
            _print_system_line(name, qsys, indent="    ")


def main() -> int:
    configure_stdout()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    memory = EpisodicMemory.create(use_clip=args.use_clip, enable_dreaming=False)
    encoder = "mock" if memory.uses_mock_encoder else "clip"
    if args.use_clip and encoder == "mock":
        print("[completion] --use-clip demandé mais CLIP indisponible, fallback mock.")

    visits = distinct_visits()
    hcann = HcannStore(memory)
    rag = RagStore(memory)
    ca3 = Ca3Store(memory)
    knn = DgKnnStore(memory)
    lexical = LexicalStore()
    n_ep = ingest_visits(hcann, visits)
    ingest_visits(lexical, visits)

    clip_rows = evaluate_visits(
        {"rag": rag, "lexical": lexical, "dg_knn": knn, "ca3": ca3, "hcann": hcann},
        visits,
        k=args.k,
        queries_fn=completion_queries,
    )
    clip_summary = compare_summaries(clip_rows)
    pattern_rows = evaluate_pattern_cues(memory, visits, k=args.k)
    pattern_summary = compare_summaries(pattern_rows)

    if encoder == "clip":
        clip_summary["note"] = (
            "Cue CLIP (crop / texte partiel / image bruitée) : ce n'est pas un "
            "fragment du motif CA3. Hopfield n'a rien à compléter."
        )
        ca3_hit = pattern_summary.get("systems", {}).get("ca3", {}).get("hit", 0.0)
        knn_hit = pattern_summary.get("systems", {}).get("dg_knn", {}).get("hit", 0.0)
        pattern_summary["note"] = (
            f"Cue = DG stocké (bruit, masque, mélange 60/40). Hit CA3 {100 * ca3_hit:.0f}% vs "
            f"k-NN DG {100 * knn_hit:.0f}%. Statut : attracteur DG, pas mieux que le cosine dans le même espace."
        )
    else:
        clip_summary["note"] = "Encodeur mock : diagnostic CLIP non visuel."
        pattern_summary["note"] = "Encodeur mock : la dynamique CA3 reste mesurable sur les motifs DG."

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "unit": "episodic_completion",
        "encoder": encoder,
        "k": args.k,
        "seed": args.seed,
        "n_episodes": n_ep,
        "clip_cue": clip_summary,
        "pattern_cue": pattern_summary,
        "summary": pattern_summary,
        "rows": pattern_rows,
        "clip_rows": clip_rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 64)
    print("  Complétion — cue CLIP vs motif DG stocké")
    print("=" * 64)
    print(f"  Encodeur     : {encoder}")
    print(f"  Épisodes     : {n_ep}")
    _print_block("  [A] Cue CLIP (mauvais test pour Hopfield)", clip_summary, ("rag", "lexical", "dg_knn", "ca3", "hcann"))
    print(f"  {clip_summary['note']}")
    _print_block("  [B] Cue = DG stocké altéré (test CA3)", pattern_summary, ("dg_knn", "ca3"))
    print(f"  {pattern_summary['note']}")
    print(f"  Export : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
