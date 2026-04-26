import torch, numpy as np
from utils.config import HCANNConfig
from utils.data_utils import get_dataloader
from models.hcann import HCANN
from train.continual_learner import ContinualLearner
from train.metrics import ContinualMetrics

def run_hcann(cfg, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_dataloader("./data/synthetic_multimodal", cfg.batch_size)
    model = HCANN(cfg).to(device)
    learner = ContinualLearner(model, cfg, device)
    metrics = ContinualMetrics(1)
    
    for ep in range(cfg.epochs_per_task):
        for b in loader:
            learner.train_step(b.get('images').to(device), b.get('texts'))
        acc = learner.evaluate(loader)
        metrics.update(0, [acc])
    return metrics.average_accuracy(), metrics.forgetting_measure()

if __name__ == "__main__":
    cfg = HCANNConfig()
    acc, forget = run_hcann(cfg)
    print(f"🔹 HCANN | Acc: {acc:.3f} | Forget: {forget:.3f}")