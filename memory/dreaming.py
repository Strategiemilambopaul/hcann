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
        memory_manager=None,
        episodes=None,
        schema_builder=None,
        llm_callback=None,
    ) -> Dict:
        n_cycles = n_cycles or config.dream_cycles
        batch_size = config.dream_batch_size
        device = next(model.parameters()).device
        episodic = bool(getattr(config, "episodic_mode", False))

        stats = {
            "patterns_replayed": 0,
            "hebb_delta_norm": 0.0,
            "stdp_updates": 0,
            "novelty_rate": 0.0,
            "evicted": 0,
            "forced_evictions": 0,
            "reconciled": {},
        }

        if episodic and memory_manager is not None:
            stats["reconciled"] = memory_manager.reconcile(
                graph.graph, model.ca3, episodes
            )
            stats["forced_evictions"] = int(memory_manager.forced_evictions)

        novelty_sum = 0.0
        novelty_count = 0

        for _ in range(n_cycles):
            batch = buffer.sample(batch_size)
            if batch is None:
                continue

            hpc_batch = batch["hpc"]
            sem_batch = batch["sem"]

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

        seq_len = int(getattr(config, "dream_seq_len", 4))
        n_seq = int(getattr(config, "dream_n_seq", n_cycles))
        sequences = []
        if hasattr(buffer, "sample_sequences"):
            sequences = buffer.sample_sequences(n_seq, seq_len)
        for window in sequences:
            ids = [item.get("meta", {}).get("episode_id") for item in window]
            for j in range(len(ids) - 1):
                if not ids[j] or not ids[j + 1] or ids[j] == ids[j + 1]:
                    continue
                graph.graph.update_hebbian_weights(ids[j], ids[j + 1], config.stdp_lr)
                stats["stdp_updates"] += 1
                if episodic and hasattr(graph, "link_next"):
                    graph.link_next(ids[j], ids[j + 1], config.stdp_lr)

        if stats["patterns_replayed"] > 0:
            stats["hebb_delta_norm"] /= stats["patterns_replayed"]
        if novelty_count > 0:
            stats["novelty_rate"] = novelty_sum / novelty_count

        graph.consolidate(llm_callback=llm_callback)
        if hasattr(graph.graph, "apply_hebbian_decay"):
            graph.graph.apply_hebbian_decay()

        if episodic and schema_builder is not None:
            created = schema_builder.maybe_build(graph.graph, llm_callback=llm_callback)
            stats["schemas_created"] = len(created)

        if episodic and memory_manager is not None:
            evicted = memory_manager.enforce(graph.graph, model.ca3, episodes)
            stats["evicted"] = len(evicted)
            stats["forced_evictions"] = int(memory_manager.forced_evictions)
        return stats
