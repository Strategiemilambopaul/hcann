import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io.wavfile import write as wav_write

# Configuration
OUTPUT_DIR = "data/synthetic_multimodal"
NUM_EPISODES = 200  # Nombre total d'épisodes à générer
NUM_TASKS = 2       # Pour split continu (ex: Task 0 = formes simples, Task 1 = formes complexes)

# Dictionnaires pour la génération
COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128)
}
SHAPES = ["circle", "square", "triangle"]
LOCATIONS = ["kitchen", "living room", "bedroom", "office"]
LABELS = list(range(len(COLORS) * len(SHAPES)))

def generate_image(color_name, shape, bg_color=(240, 240, 240)):
    """Génère une image synthétique simple"""
    img = Image.new("RGB", (64, 64), bg_color)
    draw = ImageDraw.Draw(img)
    
    color = COLORS[color_name]
    x, y = random.randint(10, 50), random.randint(10, 50)
    radius = 15
    
    if shape == "circle":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    elif shape == "square":
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)
    elif shape == "triangle":
        draw.polygon([(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)], fill=color)
        
    return img

def generate_text(color_name, shape, location):
    """Génère une description textuelle"""
    return f"Agent sees a {color_name} {shape} in the {location}."

def generate_audio(label, duration_sec=0.5, sample_rate=44100):
    """Génère un signal audio simple (onde sinusoïdale)"""
    # Fréquence dépend du label pour varier le son
    frequency = 200 + (label * 50) 
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    note = np.sin(frequency * 2 * np.pi * t)
    audio = (note * 32767).astype(np.int16)
    return sample_rate, audio

def create_dataset(num_episodes):
    """Crée le dataset complet"""
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "audio"), exist_ok=True)
    
    episodes = []
    
    print(f"🔄 Génération de {num_episodes} épisodes multimodaux...")
    
    for i in range(num_episodes):
        # 1. Sélection aléatoire des éléments
        color_name = random.choice(list(COLORS.keys()))
        shape = random.choice(SHAPES)
        location = random.choice(LOCATIONS)
        
        # Label basé sur la combinaison couleur+forme
        color_idx = list(COLORS.keys()).index(color_name)
        shape_idx = SHAPES.index(shape)
        label = color_idx * len(SHAPES) + shape_idx
        
        # 2. Génération Image
        img = generate_image(color_name, shape)
        img_path = f"images/ep_{i:04d}.png"
        img.save(os.path.join(OUTPUT_DIR, img_path))
        
        # 3. Génération Audio
        rate, audio_data = generate_audio(label)
        audio_path = f"audio/ep_{i:04d}.wav"
        wav_write(os.path.join(OUTPUT_DIR, audio_path), rate, audio_data)
        
        # 4. Génération Texte
        text_desc = generate_text(color_name, shape, location)
        
        # 5. Construction de l'épisode
        episode = {
            "id": i,
            "image_path": img_path,
            "audio_path": audio_path,
            "text": text_desc,
            "label": label,
            "metadata": {
                "color": color_name,
                "shape": shape,
                "location": location
            }
        }
        episodes.append(episode)
        
        if i % 50 == 0:
            print(f"   ↳ {i}/{num_episodes} épisodes créés...")

    # Sauvegarde JSON
    json_path = os.path.join(OUTPUT_DIR, "episodes.json")
    with open(json_path, "w") as f:
        json.dump(episodes, f, indent=2)
        
    print(f"✅ Dataset sauvegardé dans {OUTPUT_DIR}/")
    print(f"📂 Structure: {os.listdir(OUTPUT_DIR)}")

if __name__ == "__main__":
    create_dataset(NUM_EPISODES)