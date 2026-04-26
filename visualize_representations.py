import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Configuration des plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class HCANNVisualizer:
    """
    Visualiseur pour les représentations HCANN
    Analyse la séparation épisodique/sémantique via t-SNE/UMAP
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.hippo_states = []
        self.cortex_states = []
        self.labels = []
        self.task_ids = []
        self.modalities = []
        
    def extract_states(self, dataloader, max_samples: int = 2000):
        """
        Extrait les états hippocampiques et corticaux pendant l'inférence
        """
        self.model.eval()
        count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if count >= max_samples:
                    break
                    
                # Préparation des inputs
                vision = batch.get('vision')
                language = batch.get('language')
                
                if vision is not None:
                    vision = vision.to(self.device)
                
                # Encodage multimodal
                semantic_emb = self.model.multimodal_encoder(
                    vision=vision,
                    language=language,
                    mode='fusion'
                )
                
                # Passage par l'hippocampe (Vector-HaSH)
                velocity = torch.zeros(semantic_emb.size(0), 2, device=self.device)
                hpc_state, grid_states = self.model.hippocampus(
                    semantic_emb, 
                    velocity=velocity
                )
                
                # Passage par le cortex
                cortex_emb, entorhinal_emb = self.model.cortex(semantic_emb)
                
                # Stockage des états
                self.hippo_states.append(hpc_state.cpu().numpy())
                self.cortex_states.append(cortex_emb.cpu().numpy())
                
                # Métadonnées pour coloration
                if 'label' in batch:
                    self.labels.extend(batch['label'].cpu().numpy())
                if 'task_id' in batch:
                    self.task_ids.extend(batch['task_id'].cpu().numpy())
                if 'modality' in batch:
                    self.modalities.extend(batch['modality'])
                
                count += hpc_state.size(0)
                
        # Concaténation
        self.hippo_states = np.vstack(self.hippo_states)
        self.cortex_states = np.vstack(self.cortex_states)
        self.labels = np.array(self.labels) if self.labels else None
        self.task_ids = np.array(self.task_ids) if self.task_ids else None
        
        print(f"✓ États extraits: {len(self.hippo_states)} échantillons")
        print(f"  - Hippocampe: {self.hippo_states.shape}")
        print(f"  - Cortex: {self.cortex_states.shape}")
        
    def reduce_dimensionality(self, 
                            states: np.ndarray, 
                            method: str = 'umap',
                            n_components: int = 2,
                            perplexity: int = 30,
                            n_neighbors: int = 15,
                            min_dist: float = 0.1) -> np.ndarray:
        """
        Réduction de dimensionnalité pour visualisation
        """
        # Normalisation
        scaler = StandardScaler()
        states_scaled = scaler.fit_transform(states)
        
        if method == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                n_iter=1000,
                random_state=42,
                metric='cosine',
                init='pca',
                learning_rate='auto'
            )
        elif method == 'umap':
            reducer = UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric='cosine',
                random_state=42
            )
        elif method == 'pca':
            reducer = PCA(n_components=n_components)
        else:
            raise ValueError(f"Méthode inconnue: {method}")
            
        print(f"✓ Réduction {method.upper()} en cours...")
        return reducer.fit_transform(states_scaled)
    
    def plot_separation(self, 
                       hippo_2d: np.ndarray,
                       cortex_2d: np.ndarray,
                       labels: Optional[np.ndarray] = None,
                       task_ids: Optional[np.ndarray] = None,
                       title: str = "Séparation Hippocampe vs Cortex",
                       save_path: Optional[str] = None):
        """
        Visualise la séparation entre représentations hippocampiques et corticales
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Palette de couleurs
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # === Panneau 1: Hippocampe coloré par label ===
        ax = axes[0, 0]
        if labels is not None:
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(hippo_2d[mask, 0], hippo_2d[mask, 1], 
                          c=[colors[i % len(colors)]], 
                          label=f'Label {label}', 
                          s=20, alpha=0.6, edgecolors='white')
        else:
            ax.scatter(hippo_2d[:, 0], hippo_2d[:, 1], 
                      c='steelblue', s=20, alpha=0.6, edgecolors='white')
        ax.set_title("Représentations Hippocampiques (Épisodique)", fontweight='bold')
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        # === Panneau 2: Cortex coloré par label ===
        ax = axes[0, 1]
        if labels is not None:
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(cortex_2d[mask, 0], cortex_2d[mask, 1], 
                          c=[colors[i % len(colors)]], 
                          label=f'Label {label}', 
                          s=20, alpha=0.6, edgecolors='white')
        else:
            ax.scatter(cortex_2d[:, 0], cortex_2d[:, 1], 
                      c='coral', s=20, alpha=0.6, edgecolors='white')
        ax.set_title("Représentations Corticales (Sémantique)", fontweight='bold')
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        # === Panneau 3: Superposition des deux ===
        ax = axes[1, 0]
        ax.scatter(hippo_2d[:, 0], hippo_2d[:, 1], 
                  c='steelblue', label='Hippocampe', s=15, alpha=0.4)
        ax.scatter(cortex_2d[:, 0], cortex_2d[:, 1], 
                  c='coral', label='Cortex', s=15, alpha=0.4)
        ax.set_title("Superposition: Hippocampe vs Cortex", fontweight='bold')
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # === Panneau 4: Distance inter/intra-cluster ===
        ax = axes[1, 1]
        
        # Calcul des distances
        from scipy.spatial.distance import cdist
        
        if labels is not None:
            unique_labels = np.unique(labels)
            intra_hippo, intra_cortex = [], []
            inter_hippo, inter_cortex = [], []
            
            for label in unique_labels:
                mask = labels == label
                if np.sum(mask) > 1:
                    # Distance intra-cluster
                    h_dists = cdist(hippo_2d[mask], hippo_2d[mask])
                    c_dists = cdist(cortex_2d[mask], cortex_2d[mask])
                    intra_hippo.extend(h_dists[h_dists > 0])
                    intra_cortex.extend(c_dists[c_dists > 0])
                    
                    # Distance inter-cluster (vers autres labels)
                    other_mask = labels != label
                    if np.sum(other_mask) > 0:
                        h_inter = cdist(hippo_2d[mask], hippo_2d[other_mask])
                        c_inter = cdist(cortex_2d[mask], cortex_2d[other_mask])
                        inter_hippo.extend(h_inter.flatten())
                        inter_cortex.extend(c_inter.flatten())
            
            # Boxplot
            data = [intra_hippo, inter_hippo, intra_cortex, inter_cortex]
            labels_plot = ['Intra-Hippo', 'Inter-Hippo', 'Intra-Cortex', 'Inter-Cortex']
            ax.boxplot(data, labels=labels_plot, patch_artist=True)
            ax.set_title("Distances: Intra vs Inter Cluster", fontweight='bold')
            ax.set_ylabel("Distance Euclidienne")
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Figure sauvegardée: {save_path}")
        plt.show()
        
    def plot_evolution(self, 
                      states_history: List[np.ndarray],
                      labels_history: List[np.ndarray],
                      method: str = 'umap',
                      save_path: Optional[str] = None):
        """
        Visualise l'évolution des représentations pendant l'entraînement
        """
        n_epochs = len(states_history)
        fig, axes = plt.subplots(2, (n_epochs + 1) // 2, figsize=(5 * ((n_epochs + 1) // 2), 10))
        if n_epochs == 1:
            axes = [axes]
        axes = axes.flatten()
        
        for epoch, (states, labels) in enumerate(zip(states_history, labels_history)):
            reduced = self.reduce_dimensionality(states, method=method)
            
            ax = axes[epoch]
            if labels is not None:
                unique_labels = np.unique(labels)
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(reduced[mask, 0], reduced[mask, 1], 
                              s=15, alpha=0.6, label=f'{label}', edgecolors='white')
            ax.set_title(f"Epoch {epoch + 1}")
            ax.set_xticks([])
            ax.set_yticks([])
            if epoch == 0:
                ax.legend(fontsize=6, loc='best')
                
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def compute_metrics(self, 
                       hippo_2d: np.ndarray, 
                       cortex_2d: np.ndarray,
                       labels: np.ndarray) -> Dict[str, float]:
        """
        Calcule des métriques quantitatives de séparation
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        
        metrics = {}
        
        # Silhouette Score (plus élevé = meilleure séparation)
        if len(np.unique(labels)) > 1:
            metrics['silhouette_hippo'] = silhouette_score(hippo_2d, labels)
            metrics['silhouette_cortex'] = silhouette_score(cortex_2d, labels)
            
            # Davies-Bouldin Index (plus bas = meilleure séparation)
            metrics['davies_bouldin_hippo'] = davies_bouldin_score(hippo_2d, labels)
            metrics['davies_bouldin_cortex'] = davies_bouldin_score(cortex_2d, labels)
            
            # Distance moyenne inter-cluster
            unique_labels = np.unique(labels)
            inter_dists_hippo, inter_dists_cortex = [], []
            
            for i, l1 in enumerate(unique_labels):
                for l2 in unique_labels[i+1:]:
                    mask1, mask2 = labels == l1, labels == l2
                    if np.sum(mask1) > 0 and np.sum(mask2) > 0:
                        from scipy.spatial.distance import cdist
                        d_h = cdist(hippo_2d[mask1], hippo_2d[mask2]).mean()
                        d_c = cdist(cortex_2d[mask1], cortex_2d[mask2]).mean()
                        inter_dists_hippo.append(d_h)
                        inter_dists_cortex.append(d_c)
            
            if inter_dists_hippo:
                metrics['inter_cluster_dist_hippo'] = np.mean(inter_dists_hippo)
                metrics['inter_cluster_dist_cortex'] = np.mean(inter_dists_cortex)
                
        print("\n📊 Métriques de Séparation:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Visualisation HCANN - t-SNE/UMAP')
    parser.add_argument('--data_dir', type=str, default='./data/synthetic_multimodal')
    parser.add_argument('--method', type=str, choices=['tsne', 'umap', 'pca'], default='umap')
    parser.add_argument('--output_dir', type=str, default='./results/visualizations')
    parser.add_argument('--max_samples', type=int, default=2000)
    parser.add_argument('--n_neighbors', type=int, default=15)
    parser.add_argument('--min_dist', type=float, default=0.1)
    args = parser.parse_args()
    
    print("🔬 Initialisation du visualiseur HCANN...")
    
    # Import du modèle (adapte selon ta structure)
    from utils.config import HCANNConfig
    from utils.data_utils import create_multimodal_loader
    from models.hcann import HCANN
    
    # Configuration
    cfg = HCANNConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Chargement des données
    print("📦 Chargement du dataset...")
    loader = create_multimodal_loader(
        data_dir=args.data_dir,
        modalities=['vision', 'language'],
        batch_size=32,
        num_tasks=2
    )
    
    # Initialisation du modèle
    print("🧠 Chargement du modèle HCANN...")
    model = HCANN(cfg).to(device)
    
    # [Optionnel] Charger des poids entraînés
    # model.load_state_dict(torch.load('checkpoints/hcann_best.pt', map_location=device))
    
    # Visualisation
    visualizer = HCANNVisualizer(model, device)
    
    print("🔍 Extraction des états...")
    visualizer.extract_states(loader[0], max_samples=args.max_samples)
    
    print(f"📉 Réduction de dimension ({args.method.upper()})...")
    hippo_2d = visualizer.reduce_dimensionality(
        visualizer.hippo_states, 
        method=args.method,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )
    cortex_2d = visualizer.reduce_dimensionality(
        visualizer.cortex_states,
        method=args.method,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )
    
    print("🎨 Génération des visualisations...")
    visualizer.plot_separation(
        hippo_2d, cortex_2d,
        labels=visualizer.labels,
        task_ids=visualizer.task_ids,
        title=f"HCANN: Séparation Épisodique/Sémantique ({args.method.upper()})",
        save_path=f"{args.output_dir}/separation_{args.method}.png"
    )
    
    # Métriques quantitatives
    if visualizer.labels is not None:
        metrics = visualizer.compute_metrics(hippo_2d, cortex_2d, visualizer.labels)
        
        # Sauvegarde des métriques
        import json
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{args.output_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
    
    print("\n✅ Visualisation terminée !")


if __name__ == "__main__":
    main()