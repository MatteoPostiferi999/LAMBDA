import os
import torch
import logging
import subprocess
from huggingface_hub import snapshot_download
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
import trimesh

logging.basicConfig(level=logging.INFO)

# Dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Usando device: {device}")

# Setup ambiente (se non ancora fatto)
if not os.path.exists("Hunyuan3D-2GP"):
    subprocess.run(["git", "clone", "https://github.com/deepbeepmeep/Hunyuan3D-2GP.git"])
os.chdir("Hunyuan3D-2GP")

subprocess.run(["sed", "-i", "s/mmgp/hy3dgen/g", "gradio_app.py"])
subprocess.run(["sed", "-i", "s/mmgp/hy3dgen/g", "hy3dgen/shapegen/models/autoencoders/surface_extractors.py"])
subprocess.run(["sed", "-i", "/from hy3dgen import offload/d", "hy3dgen/shapegen/models/autoencoders/surface_extractors.py"])

os.chdir("hy3dgen/texgen/custom_rasterizer")
subprocess.run(["python3", "setup.py", "install"])
os.chdir("../../differentiable_renderer")
subprocess.run(["python3", "setup.py", "install"])
os.chdir("../../")

# Scarica modelli necessari per Hunyuan3DPaintPipeline
local_dir = snapshot_download(
    repo_id="tencent/Hunyuan3D-2",
    allow_patterns=[
        "hunyuan3d-delight-v2-0/**",
        "hunyuan3d-paint-v2-0/**"
    ],
    resume_download=True
)
print("📦 Modello Paint scaricato in:", local_dir)

# Crea la pipeline di texturing
pipeline = Hunyuan3DPaintPipeline.from_pretrained(local_dir)

# Carica la mesh salvata (es. generata da DiTFlowMatching)

if not os.path.exists(mesh_path):
    raise FileNotFoundError(f"⚠️ Mesh non trovata: {mesh_path}")

mesh_path = input("Inserisci il percorso della mesh (.ply): ").strip()
mesh = trimesh.load(mesh_path)
print("✅ Mesh caricata:", mesh_path)

# Carica immagine da usare come texture

if not os.path.exists(image_path):
    raise FileNotFoundError(f"⚠️ Immagine non trovata: {image_path}")

image_path = input("Inserisci il percorso dell'immagine da usare come texture (.png/.jpg): ").strip()
print("🖼️ Immagine caricata:", image_path)

# Applica la texture
with torch.no_grad():
    mesh_textured = pipeline(mesh, image=image_path)

# Esporta la mesh texturizzata
output_path = "textured_output.ply"
mesh_textured.export(output_path)
print("✅ Mesh texturizzata salvata in:", output_path)
