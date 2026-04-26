import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, json

class MultimodalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        with open(os.path.join(data_dir, 'episodes.json')) as f:
            self.episodes = json.load(f)
            
    def __len__(self): return len(self.episodes)
    def __getitem__(self, idx):
        ep = self.episodes[idx]
        img = Image.open(os.path.join(self.data_dir, ep['image_path'])).convert('RGB')
        if self.transform: img = self.transform(img)
        return {'images': img, 'texts': [ep['text']], 'labels': ep['label']}

def get_dataloader(data_dir, batch_size=32, shuffle=True):
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    ds = MultimodalDataset(data_dir, tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)