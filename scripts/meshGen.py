import os
import torch
import logging
import subprocess
from huggingface_hub import snapshot_download
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import trimesh
from PIL import Image


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


# Determina se il modello selezionato è multiview
is_multiview = 'mv' in subfolder.lower()

if is_multiview:
    print("🔄 Modello multiview selezionato. Fornisci le immagini per le viste disponibili.")
    image_dict = {}
    for view in ['front', 'left', 'right', 'back']:
        path = input(f"Inserisci il percorso dell'immagine per la vista '{view}' (lascia vuoto per saltare): ").strip()
        if path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ Immagine non trovata: {path}")
            image_dict[view] = Image.open(path).convert("RGB")
    if not image_dict:
        raise ValueError("❌ Nessuna immagine fornita per il modello multiview.")
    image_input = image_dict
else:
    image_path = input("Inserisci il percorso dell'immagine (es. demo.png): ").strip()
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Immagine non trovata: {image_path}")
    image_input = Image.open(image_path).convert("RGB")



# Genera mesh
with torch.no_grad():
    mesh = pipeline(
        image=image_input,
        target_face_number=40000,
        inference_steps=42,
        seed=42,
        octree_resolution=256
    )[0]

# Esporta la mesh
output_path = "output/output_custom.ply"

# Assicurati che esista la directory di output
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Esporta la mesh
mesh.export(output_path)
print(f"✅ Mesh generata e salvata in: {output_path}")
