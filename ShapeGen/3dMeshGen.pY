import os
import torch
import logging
import subprocess
from huggingface_hub import snapshot_download
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import trimesh

logging.basicConfig(level=logging.INFO)

# Dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Usando device: {device}")

# Funzione di scelta modello
def select_hunyuan_model():
    models = {
        '1': {'name': 'hunyuan3d-dit-v2-mini',      'repo': 'tencent/Hunyuan3D-2mini', 'subfolder': 'hunyuan3d-dit-v2-mini'},
        '2': {'name': 'hunyuan3d-dit-v2-mini-fast', 'repo': 'tencent/Hunyuan3D-2mini', 'subfolder': 'hunyuan3d-dit-v2-mini-fast'},
        '3': {'name': 'hunyuan3d-dit-v2-mini-turbo','repo': 'tencent/Hunyuan3D-2mini', 'subfolder': 'hunyuan3d-dit-v2-mini-turbo'},
        '4': {'name': 'hunyuan3d-dit-v2-0',         'repo': 'tencent/Hunyuan3D-2',     'subfolder': 'hunyuan3d-dit-v2-0'},
        '5': {'name': 'hunyuan3d-dit-v2-0-fast',    'repo': 'tencent/Hunyuan3D-2',     'subfolder': 'hunyuan3d-dit-v2-0-fast'},
        '6': {'name': 'hunyuan3d-dit-v2-0-turbo',   'repo': 'tencent/Hunyuan3D-2',     'subfolder': 'hunyuan3d-dit-v2-0-turbo'},
        '7': {'name': 'hunyuan3d-dit-v2-mv',        'repo': 'tencent/Hunyuan3D-2mv',   'subfolder': 'hunyuan3d-dit-v2-mv'},
        '8': {'name': 'hunyuan3d-dit-v2-mv-fast',   'repo': 'tencent/Hunyuan3D-2mv',   'subfolder': 'hunyuan3d-dit-v2-mv-fast'},
        '9': {'name': 'hunyuan3d-dit-v2-mv-turbo',  'repo': 'tencent/Hunyuan3D-2mv',   'subfolder': 'hunyuan3d-dit-v2-mv-turbo'},
    }
    print("📦 1–3: mini    4–6: full-scale    7–9: multi-view")
    for k, v in models.items():
        print(f" {k}. {v['name']}")
    choice = None
    while choice not in models:
        choice = input("Modello (1–9)? ").strip()
    return models[choice]

# Scegli modello
model_info = select_hunyuan_model()
repo_id    = model_info['repo']
subfolder  = model_info['subfolder']
print(f"⏬ Scarico da {repo_id}, sotto-cartella {subfolder}…")

# Scarica il contenuto nella cache locale
local_dir = snapshot_download(
    repo_id=repo_id,
    allow_patterns=[
        f"{subfolder}/config.yaml",
        f"{subfolder}/*.safetensors"
    ],
    resume_download=True
)

print("→ Contenuto scaricato in:", local_dir)

# Crea la pipeline
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    local_dir,
    subfolder=subfolder,
    device=device,
    local_files_only=True
)

# Carica immagine da path locale (richiesto all'utente)
image_path = input("Inserisci il percorso dell'immagine (es. demo.png): ").strip()

# Genera mesh
with torch.no_grad():
    mesh = pipeline(
        image=image_path,
        target_face_number=20000,
        inference_steps=20,
        seed=42,
        octree_resolution=128
    )[0]

# Esporta la mesh
output_path = 'output_custom.ply'
mesh.export(output_path)
print(f"✅ Mesh generata e salvata in: {output_path}")
