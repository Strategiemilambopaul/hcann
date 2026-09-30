import torch
from typing import Dict, Optional


class DreamingPhase:
    """Phase sommeil : rejeu offline, Hebb Subiculum, STDP graphe — sans backprop."""

    @staticmethod
    @torch.no_grad()
    def run_sleep_cycle(
        buffer,
        model,
        graph,
        config,
        n_cycles: Optional[int] = None,
    ) -> Dict:
        n_cycles = n_cycles or config.dream_cycles
        batch_size = config.dream_batch_size
        device = next(model.parameters()).device

        stats = {
            "patterns_replayed": 0,
            "hebb_delta_norm": 0.0,
            "stdp_updates": 0,
            "novelty_rate": 0.0,
        }

        novelty_sum = 0.0
        novelty_count = 0

        for _ in range(n_cycles):
            batch = buffer.sample(batch_size)
            if batch is None:
                continue

            hpc_batch = batch["hpc"]
            sem_batch = batch["sem"]
            meta_batch = batch.get("meta", [])

            for i in range(hpc_batch.size(0)):
                hpc = hpc_batch[i].to(device)
                sem = sem_batch[i].to(device)

                completed = model.ca3.complete(hpc)
                if completed.dim() == 1:
                    completed = completed.unsqueeze(0)
                hpc_vec = completed.squeeze(0)
                delta = model.local_hebb_update(sem, hpc_vec)
                stats["hebb_delta_norm"] += delta
                stats["patterns_replayed"] += 1
                model.ca1.hebb_update(hpc_vec, sem)

                _, novelty, _ = model.ca1(completed, sem.unsqueeze(0))
                novelty_sum += novelty.mean().item()
                novelty_count += 1

            if len(meta_batch) >= 2:
                ids = [m.get("episode_id") for m in meta_batch if m.get("episode_id")]
                for j in range(len(ids) - 1):
                    if ids[j] and ids[j + 1]:
                        graph.graph.update_hebbian_weights(
                            ids[j], ids[j + 1], config.stdp_lr
                        )
                        stats["stdp_updates"] += 1

        if stats["patterns_replayed"] > 0:
            stats["hebb_delta_norm"] /= stats["patterns_replayed"]
        if novelty_count > 0:
            stats["novelty_rate"] = novelty_sum / novelty_count

        graph.consolidate(llm_callback=None)
        if hasattr(graph.graph, "apply_hebbian_decay"):
            graph.graph.apply_hebbian_decay()
        return stats
