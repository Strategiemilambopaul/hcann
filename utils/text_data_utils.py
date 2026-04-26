import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import numpy as np

class TextEpisodicDataset(Dataset):
    """Dataset texte → embeddings fixes pour HCANN"""
    def __init__(self, texts, labels, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        # Encodage batché
        self.embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        self.labels = np.array(labels)
        
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def create_incremental_text_loader(texts, labels, num_tasks=3, batch_size=32):
    """
    Split incrémental par thèmes.
    Ex: Task 0 = Tech, Task 1 = Sport, Task 2 = Politique
    """
    unique_labels = np.unique(labels)
    assert len(unique_labels) >= num_tasks, "Pas assez de classes pour le nombre de tâches"
    
    loaders = []
    classes_per_task = len(unique_labels) // num_tasks
    
    for t in range(num_tasks):
        start_cls = unique_labels[t * classes_per_task]
        end_cls = unique_labels[(t + 1) * classes_per_task] if t < num_tasks - 1 else unique_labels[-1] + 1
        
        mask = (labels >= start_cls) & (labels < end_cls)
        subset_texts = [texts[i] for i in range(len(texts)) if mask[i]]
        subset_labels = labels[mask]
        
        dataset = TextEpisodicDataset(subset_texts, subset_labels)
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))
        
    return loaders