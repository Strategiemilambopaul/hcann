import torch
from utils.config import HCANNConfig
from utils.data_utils import get_dataloader
from models.hcann import HCANN
from train.continual_learner import ContinualLearner
from train.metrics import ContinualMetrics

def main():
    cfg = HCANNConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 HCANN Multimodal sur {device}")
    
    loader = get_dataloader("./data/synthetic_multimodal", cfg.batch_size)
    model = HCANN(cfg).to(device)
    learner = ContinualLearner(model, cfg, device)
    metrics = ContinualMetrics(num_tasks=1)
    
    for epoch in range(cfg.epochs_per_task):
        for batch in loader:
            learner.train_step(
                images=batch.get('images').to(device),
                texts=batch.get('texts'),
                velocity=torch.zeros(batch['images'].size(0), 2, device=device)
            )
        acc = learner.evaluate(loader)
        print(f"Epoch {epoch+1} | Acc: {acc:.4f}")
        metrics.update(0, [acc])
        
    print(f"\n📊 Average Accuracy: {metrics.average_accuracy():.4f}")
    print(f"📉 Forgetting: {metrics.forgetting_measure():.4f}")

if __name__ == "__main__":
    main()