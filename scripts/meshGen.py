import os
import torch
import logging
from datetime import datetime
from huggingface_hub import snapshot_download
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from PIL import Image
import trimesh

logging.basicConfig(level=logging.INFO)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"🖥️ Usando device: {device}")

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

def download_model_and_create_pipeline():
    model_info = select_hunyuan_model()
    repo_id = model_info['repo']
    subfolder = model_info['subfolder']

    logging.info(f"⏬ Scarico da {repo_id}, sotto-cartella {subfolder}…")
    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[
            f"{subfolder}/config.yaml",
            f"{subfolder}/*.safetensors"
        ],
        resume_download=True
    )
    logging.info(f"→ Contenuto scaricato in: {local_dir}")

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        local_dir,
        subfolder=subfolder,
        device=device,
        local_files_only=True
    )
    return pipeline, subfolder

def get_image_input(subfolder):
    is_multiview = 'mv' in subfolder.lower()
    if is_multiview:
        logging.info("🔄 Modello multiview selezionato. Fornisci le immagini per le viste disponibili.")
        image_dict = {}
        for view in ['front', 'left', 'right', 'back']:
            path = input(f"Immagine per la vista '{view}' (lascia vuoto per saltare): ").strip()
            if path:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"❌ Immagine non trovata: {path}")
                image_dict[view] = Image.open(path).convert("RGB")
        if not image_dict:
            raise ValueError("❌ Nessuna immagine fornita per il modello multiview.")
        return image_dict
    else:
        path = input("Inserisci il percorso dell'immagine (es. demo.png): ").strip()
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Immagine non trovata: {path}")
        return Image.open(path).convert("RGB")

def generate_and_save_mesh(pipeline, image_input):
    with torch.no_grad():
        mesh = pipeline(
            image=image_input,
            target_face_number=40000,
            inference_steps=42,
            seed=42,
            octree_resolution=256
        )[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/output_{timestamp}.ply"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh.export(output_path)
    logging.info(f"✅ Mesh generata e salvata in: {output_path}")
    return output_path

if __name__ == "__main__":
    pipeline, subfolder = download_model_and_create_pipeline()
    image_input = get_image_input(subfolder)
    generate_and_save_mesh(pipeline, image_input)
