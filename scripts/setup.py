import os
import subprocess
import sys

# Configurazione iniziale
USE_CUDA = True
INSTALL_FLAGS = ["--user"]  # Installa i moduli localmente (senza permessi root)

# 1. Clona il repository
if not os.path.exists("Hunyuan3D-2GP"):
    subprocess.run(["git", "clone", "https://github.com/deepbeepmeep/Hunyuan3D-2GP.git"])

os.chdir("Hunyuan3D-2GP")

# 2. Downgrade di NumPy per compatibilità con PyTorch e moduli compilati
subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4"])

# 3. Installa PyTorch (CUDA se possibile)
if USE_CUDA:
    subprocess.run([
        sys.executable, "-m", "pip", "install", "torch==2.5.1", "torchvision", "torchaudio"
    ] + INSTALL_FLAGS)
else:
    subprocess.run([
        sys.executable, "-m", "pip", "install", "torch==2.5.1+cpu", "torchvision", "torchaudio"
    ] + INSTALL_FLAGS)

# 4. Installa requirements e dipendenze extra
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"] + INSTALL_FLAGS)
subprocess.run([sys.executable, "-m", "pip", "install",
                "open3d", "accelerate", "omegaconf", "Pillow", "einops", "pymeshlab", "rembg", "onnxruntime", "trimesh"
                ] + INSTALL_FLAGS)

# 5. Patch file sorgente se serve
subprocess.run(["sed", "-i", "s/mmgp/hy3dgen/g", "gradio_app.py"])
subprocess.run(["sed", "-i", "s/mmgp/hy3dgen/g", "hy3dgen/shapegen/models/autoencoders/surface_extractors.py"])
subprocess.run(["sed", "-i", "/from hy3dgen import offload/d", "hy3dgen/shapegen/models/autoencoders/surface_extractors.py"])

# 6. Compila moduli custom
custom_rasterizer_path = "hy3dgen/texgen/custom_rasterizer"
if os.path.exists(custom_rasterizer_path):
    os.chdir(custom_rasterizer_path)
    subprocess.run([sys.executable, "setup.py", "install"] + INSTALL_FLAGS)

# 7. (Facoltativo) Compila differenziable_renderer solo se esiste
diff_renderer_path = "../../differentiable_renderer"
if os.path.exists(diff_renderer_path):
    os.chdir(diff_renderer_path)
    subprocess.run([sys.executable, "setup.py", "install"] + INSTALL_FLAGS)

# 8. Torna alla root del progetto
os.chdir("../../..")
print("✅ Setup completato con successo.")
