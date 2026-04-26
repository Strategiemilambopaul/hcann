import torch
import numpy as np
from utils.config import HCANNConfig
from utils.text_data_utils import create_incremental_text_loader
from models.hcann import HCANN
from memory.consolidation import ConsolidationManager
from train import ContinualLearner, ContinualMetrics

def main():
    cfg = HCANNConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🌍 HCANN pour Texte | Device: {device}")

    # 📚 Données exemple (remplace par ton corpus)
    sample_texts = [
        "L'intelligence artificielle transforme la recherche médicale.",
        "Le match de football s'est terminé sur un score nul.",
        "Les élections municipales auront lieu en mars.",
        "Les nouveaux processeurs quantiques promettent une révolution.",
        "L'équipe nationale se prépare pour la coupe du monde.",
        "Le parlement vote la nouvelle loi sur le climat.",
        "La médecine personnalisée utilise le séquençage ADN.",
        "Le tennisman remporte son troisième grand chelem.",
        "La bourse enregistre une hausse historique.",
        "Les négociations diplomatiques reprennent à Genève."
    ]
    # Labels 0 à 9 (10 topics simulés)
    sample_labels = list(range(10)) * 2  # 20 samples

    # 🔀 Split incrémental (3 tâches : 0-2, 3-5, 6-9)
    loaders = create_incremental_text_loader(sample_texts, np.array(sample_labels), num_tasks=3, batch_size=cfg.batch_size)
    
    # 🧠 Modèle
    model = HCANN(cfg).to(device)
    
    # 🕸️ Consolidation Hebbienne
    consolidation = ConsolidationManager(cfg, device)
    
    # 🎓 Learner
    learner = ContinualLearner(model, cfg, device)
    learner.consolidation_manager = consolidation  # Liaison explicite
    metrics = ContinualMetrics(num_tasks=3)

    # 🔄 Boucle d'apprentissage continu
    for task_id, train_loader in enumerate(loaders):
        print(f"\n📖 === Tâche Texte {task_id + 1} ===")
        model.train()
        
        for epoch in range(cfg.epochs_per_task):
            total_loss = 0.0
            for x_emb, y in train_loader:
                x_emb, y = x_emb.to(device), y.to(device)
                loss = learner.train_step(x_emb, y, task_id)
                total_loss += loss
            print(f"  ↳ Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

        # 📊 Évaluation sur toutes les tâches vues
        task_accs = []
        for eval_id, eval_loader in enumerate(loaders):
            acc = learner.evaluate(eval_loader)
            task_accs.append(acc)
            print(f"  ↳ Recall Tâche {eval_id+1}: {acc:.4f}")
            
        metrics.update(task_id, task_accs)

    # 📈 Résultats finaux
    print("\n📊 Métriques Finales (Texte)")
    print(f"  • Average Accuracy   : {metrics.average_accuracy():.4f}")
    print(f"  • Forgetting Measure : {metrics.forgetting_measure():.4f}")
    print(f"  • Backward Transfer  : {metrics.backward_transfer():.4f}")

if __name__ == "__main__":
    main()