import os
import subprocess
import sys

# 1. Clona il repository
if not os.path.exists("Hunyuan3D-2GP"):
    subprocess.run(["git", "clone", "https://github.com/deepbeepmeep/Hunyuan3D-2GP.git"])
os.chdir("Hunyuan3D-2GP")

# 2. Installa PyTorch compatibile (modifica versione se necessario)
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "torch==2.5.1+cu124", "torchvision", "torchaudio"
])

# 3. Installa le altre dipendenze
subprocess.run([
    sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
])
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "open3d", "accelerate", "omegaconf", "Pillow", "einops", "pymeshlab", "rembg", "onnxruntime"
])

# 4. Sistema i nomi dei moduli se serve
subprocess.run(["sed", "-i", "s/mmgp/hy3dgen/g", "gradio_app.py"])
subprocess.run(["sed", "-i", "s/mmgp/hy3dgen/g", "hy3dgen/shapegen/models/autoencoders/surface_extractors.py"])

# 5. Installa i moduli custom
os.chdir("hy3dgen/texgen/custom_rasterizer")
subprocess.run([sys.executable, "setup.py", "install"])
os.chdir("../../differentiable_renderer")
subprocess.run([sys.executable, "setup.py", "install"])
os.chdir("../../../")  # Torna alla root del progetto

# 6. Controlla e installa trimesh / pymeshlab se mancanti
try:
    import trimesh
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "trimesh"])

try:
    import pymeshlab
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "pymeshlab"])

# 7. Rimuove eventuali import obsoleti
subprocess.run(["sed", "-i", "/from hy3dgen import offload/d", "hy3dgen/shapegen/models/autoencoders/surface_extractors.py"])
