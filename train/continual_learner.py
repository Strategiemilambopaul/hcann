import torch
from models.hcann import HCANN
from memory.consolidation import HebbianConsolidation
from memory.replay_buffer import EpisodicBuffer
from memory.dreaming import DreamingPhase


class ContinualLearner:
    def __init__(self, model: HCANN, config, device, track_baselines: bool = False):
        self.model = model
        self.config = config
        self.device = device
        self.graph = HebbianConsolidation(config, device)
        self.buffer = EpisodicBuffer()
        self.step_count = 0
        self._prev_episode_id: str | None = None
        self.track_baselines = track_baselines

    def train_step(
        self,
        images=None,
        texts=None,
        velocity=None,
        episode_id=None,
        metadata=None,
    ):
        """Encodage rapide -> graphe hebbien -> dreaming périodique (sans backprop)."""
        self.model.train()

        hpc, sem, wm, novelty, dg_code, _ = self.model.fast_encode(
            images, texts, velocity, episode_keys=episode_id
        )

        batch_size = hpc.size(0)
        for i in range(batch_size):
            hpc_np = hpc[i].detach().cpu().numpy()
            sem_np = sem[i].detach().cpu().numpy()
            dg_np = dg_code[i].detach().cpu().numpy()
            retrieved = self.graph.retrieve(hpc_np, k=3)
            current_id = str(episode_id[i]) if episode_id else f"train_{self.step_count}"
            node_data = {"step": self.step_count}
            if metadata and i < len(metadata):
                node_data.update(metadata[i])
            if texts is not None:
                text_i = texts[i] if isinstance(texts, (list, tuple)) else texts
                node_data["text"] = text_i
            node_data["novelty"] = novelty[i].item() if novelty.numel() else 0.0
            self.graph.add_node(
                current_id, node_data, hpc_np, sem_embedding=sem_np, dg_embedding=dg_np
            )

            related_ids = [r.get("id") for r in retrieved if r.get("id")]
            if related_ids:
                self.graph.update_connections(current_id, related_ids, strength=0.12)
            sem_retrieved = self.graph.retrieve_semantic(sem_np, k=4)
            sem_related = [
                r.get("id") for r in sem_retrieved
                if r.get("id") and str(r.get("id")) != current_id
            ]
            if sem_related:
                self.graph.update_connections(current_id, sem_related[:2], strength=0.10)
            if self._prev_episode_id and self._prev_episode_id != current_id:
                self.graph.update_connections(
                    current_id, [self._prev_episode_id], strength=0.08
                )
            self._prev_episode_id = current_id

            self.buffer.push(
                sem[i],
                hpc[i],
                metadata={
                    "step": self.step_count,
                    "episode_id": current_id,
                    "novelty": novelty[i].item() if novelty.numel() else 0.0,
                },
            )
            self.model.ca1.hebb_update(hpc[i], sem[i])
            self.step_count += 1
            if (
                self.config.enable_dreaming
                and self.step_count % self.config.consolidation_freq == 0
            ):
                DreamingPhase.run_sleep_cycle(
                    self.buffer, self.model, self.graph, self.config
                )

    def finalize_baselines(self) -> None:
        return None

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        correct, total = 0, 0
        for batch in dataloader:
            imgs, lbls = batch.get("images"), batch.get("labels")
            if imgs is not None:
                imgs = imgs.to(self.device)
            hpc, sem, _, _, _, _ = self.model.fast_encode(
                images=imgs, store=False
            )
            logits = self.model.subiculum(hpc)
            num_classes = int(lbls.max().item()) + 1
            preds = logits[:, :num_classes].argmax(dim=-1)
            correct += (preds == lbls.to(self.device)).sum().item()
            total += lbls.size(0)
        return correct / total if total > 0 else 0.0
