import torch
from models.hcann import HCANN
from memory.consolidation import HebbianConsolidation
from memory.episodes import EpisodeIndex
from memory.forgetting import MemoryManager, warn_capacity
from memory.recall import EpisodicRecall
from memory.replay_buffer import EpisodicBuffer
from memory.dreaming import DreamingPhase
from memory.schemas import SchemaBuilder


def _batch_len(images, texts, episode_id) -> int:
    if episode_id is not None:
        return len(episode_id)
    if isinstance(texts, (list, tuple)):
        if len(texts) == 1 and isinstance(texts[0], (list, tuple)):
            return len(texts[0])
        return len(texts)
    if texts is not None:
        return 1
    if images is not None and hasattr(images, "shape"):
        return int(images.shape[0])
    return 1


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
        self.episodic_mode = bool(getattr(config, "episodic_mode", False))
        self.episodes = EpisodeIndex(config) if self.episodic_mode else None
        self._prev_dg = None
        self.memory_manager = MemoryManager(config) if self.episodic_mode else None
        self.schema_builder = SchemaBuilder(config) if self.episodic_mode else None
        self._recall = (
            EpisodicRecall(model, self.graph, self.episodes, config)
            if self.episodic_mode
            else None
        )
        if self.episodic_mode:
            warn_capacity(config)
            g = self.graph.graph

            def _evict_policy():
                self.memory_manager.enforce(g, self.model.ca3, self.episodes)

            g.eviction_policy = _evict_policy
            self.model.ca3.set_eviction_handler(_evict_policy)

    def recall(self, cue_texts=None, cue_images=None, scope="segment", follow=None, touch=True, cue_dg=None):
        """Délégation mince vers EpisodicRecall (None si episodic_mode est faux)."""
        if self._recall is None:
            from memory.recall import RecallResult
            return RecallResult([], 0.0, "inconnu", {}, None, None)
        return self._recall.recall(
            cue_texts=cue_texts,
            cue_images=cue_images,
            scope=scope,
            follow=follow,
            touch=touch,
            cue_dg=cue_dg,
        )

    def recall_general(self, cue_texts=None, cue_images=None, k=1):
        if self._recall is None:
            return []
        return self._recall.recall_general(cue_texts=cue_texts, cue_images=cue_images, k=k)

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
        planned_ids = None
        episode_keys = episode_id
        if self.episodic_mode:
            n = _batch_len(images, texts, episode_id)
            planned_ids = []
            for i in range(n):
                if episode_id is not None:
                    planned_ids.append(str(episode_id[i]))
                else:
                    planned_ids.append(f"train_{self.step_count + i}")
            episode_keys = planned_ids

        hpc, sem, wm, novelty, dg_code, hpc_out = self.model.fast_encode(
            images, texts, velocity, episode_keys=episode_keys
        )

        batch_size = hpc.size(0)
        for i in range(batch_size):
            hpc_np = hpc[i].detach().cpu().numpy()
            sem_np = sem[i].detach().cpu().numpy()
            dg_np = dg_code[i].detach().cpu().numpy()
            if planned_ids is not None:
                current_id = planned_ids[i]
            else:
                current_id = str(episode_id[i]) if episode_id else f"train_{self.step_count}"

            if self.episodic_mode:
                self.graph.tick()

            retrieved = self.graph.retrieve(hpc_np, k=3, touch=False)
            sem_retrieved = self.graph.retrieve_semantic(sem_np, k=4, touch=False)
            if sem_retrieved:
                novelty_sem = 1.0 - float(sem_retrieved[0]["score"])
            else:
                novelty_sem = 0.0
            node_data = {"step": self.step_count}
            if metadata and i < len(metadata):
                node_data.update(metadata[i])
            if texts is not None:
                text_i = texts[i] if isinstance(texts, (list, tuple)) else texts
                node_data["text"] = text_i
            nov_i = novelty[i].item() if novelty.numel() else 0.0
            node_data["novelty"] = nov_i
            novelty_mem = nov_i
            if hpc_out is not None and "novelty_mem" in hpc_out:
                novelty_mem = float(hpc_out["novelty_mem"][i].item())
            signal = getattr(self.config, "seg_signal", "dg")
            if signal == "sem":
                seg_value = novelty_sem
            elif signal == "mix":
                seg_value = 0.5 * (float(novelty_mem) + novelty_sem)
            else:
                if int(getattr(self.config, "ctx_dim", 0)) > 0:
                    seg_value = novelty_sem
                else:
                    seg_value = float(novelty_mem)
            self._prev_dg = dg_np.copy()
            if self.episodes is not None:
                seg = self.episodes.add_event(current_id, seg_value)
                node_data["segment_id"] = seg["segment_id"]
                node_data["event_idx"] = seg["event_idx"]
                node_data["novelty_mem"] = novelty_mem
                node_data["novelty_sem"] = novelty_sem
                if hpc_out is not None and "ctx_code" in hpc_out:
                    node_data["ctx_code"] = hpc_out["ctx_code"][i].detach().cpu().numpy().reshape(-1)
                if seg["boundary"]:
                    self.model.hippocampus.temporal_ctx.boundary()
                    if bool(getattr(self.config, "reencode_boundary", True)) and hpc_out is not None:
                        dg_new = self.model.hippocampus.reencode_at_context(
                            hpc_out["sdr_ec"][i].detach(),
                            hpc_out["grid_code"][i].detach(),
                            current_id,
                        )
                        dg_np = dg_new[0].detach().cpu().numpy()
                        node_data["ctx_code"] = (
                            self.model.hippocampus.temporal_ctx.context.detach().cpu().numpy().reshape(-1)
                        )

            if self.episodic_mode:
                clock = int(self.graph.graph.clock)
                existing = self.graph.graph.nodes.get(current_id)
                if existing and "born" in (existing.get("data") or {}):
                    node_data["born"] = existing["data"]["born"]
                else:
                    node_data["born"] = clock
                node_data["surprise"] = float(max(0.0, min(1.0, float(seg_value))))
                node_data.setdefault("recall_count", 0)
                node_data.setdefault("last_recall", -1)
                node_data.setdefault("type", "episode")

            self.graph.add_node(
                current_id, node_data, hpc_np, sem_embedding=sem_np, dg_embedding=dg_np
            )

            related_ids = [
                r.get("id") for r in retrieved
                if r.get("id") and str(r.get("id")) != current_id
            ]
            if related_ids:
                self.graph.update_connections(current_id, related_ids, strength=0.12)
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
                if self.episodic_mode:
                    self.graph.link_next(self._prev_episode_id, current_id, 0.08)
            self._prev_episode_id = current_id

            buf_meta = {
                "step": self.step_count,
                "episode_id": current_id,
                "novelty": nov_i,
            }
            if "segment_id" in node_data:
                buf_meta["segment_id"] = node_data["segment_id"]
                buf_meta["event_idx"] = node_data["event_idx"]
                buf_meta["novelty_mem"] = node_data["novelty_mem"]
            self.buffer.push(
                sem[i],
                hpc[i],
                metadata=buf_meta,
            )
            self.model.ca1.hebb_update(hpc[i], sem[i])
            self.step_count += 1
            if (
                self.config.enable_dreaming
                and self.step_count % self.config.consolidation_freq == 0
            ):
                DreamingPhase.run_sleep_cycle(
                    self.buffer,
                    self.model,
                    self.graph,
                    self.config,
                    memory_manager=self.memory_manager,
                    episodes=self.episodes,
                    schema_builder=self.schema_builder,
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
