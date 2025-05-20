import os
import torch
import logging
import subprocess
from datetime import datetime
from huggingface_hub import snapshot_download
from hy3dgen.texgen import Hunyuan3DPaintPipeline
import trimesh

logging.basicConfig(level=logging.INFO)

class TextureGenerator:
    def __init__(self, repo_dir="Hunyuan3D-2GP", output_dir="output"):
        self.repo_dir = os.path.abspath(repo_dir)
        self.output_dir = output_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"🖥️ Usando device: {self.device}")

    def verify_repo(self):
        if not os.path.exists(self.repo_dir):
            raise FileNotFoundError("❌ Repository Hunyuan3D-2GP non trovato. Esegui prima setup_lambda.py")

    def fix_imports(self):
        subprocess.run(["sed", "-i", "s/mmgp/hy3dgen/g", os.path.join(self.repo_dir, "gradio_app.py")], check=True)
        surface_path = os.path.join(self.repo_dir, "hy3dgen/shapegen/models/autoencoders/surface_extractors.py")
        subprocess.run(["sed", "-i", "s/mmgp/hy3dgen/g", surface_path], check=True)
        subprocess.run(["sed", "-i", "/from hy3dgen import offload/d", surface_path], check=True)

    def build_module(self, module_path, name):
        try:
            subprocess.run(["python3", "setup.py", "install", "--user"], cwd=module_path, check=True)
            print(f"✅ {name} installato in: {module_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Errore installazione {name}:\n{e}")
            exit(1)

    def setup_build(self):
        self.fix_imports()
        rasterizer_path = os.path.join(self.repo_dir, "hy3dgen", "texgen", "custom_rasterizer")
        renderer_path = os.path.join(self.repo_dir, "hy3dgen", "differentiable_renderer")
        self.build_module(rasterizer_path, "Rasterizer")
        self.build_module(renderer_path, "Differentiable Renderer")

    def download_model(self):
        model_dir = snapshot_download(
            repo_id="tencent/Hunyuan3D-2",
            allow_patterns=[
                "hunyuan3d-delight-v2-0/**",
                "hunyuan3d-paint-v2-0/**"
            ],
            resume_download=True
        )
        print("📦 Modello Paint scaricato in:", model_dir)
        return model_dir

    def apply_texture(self, mesh_path: str, image_path: str) -> str:
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"⚠️ Mesh non trovata: {mesh_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"⚠️ Immagine non trovata: {image_path}")

        mesh = trimesh.load(mesh_path)
        logging.info(f"✅ Mesh caricata: {mesh_path}")
        logging.info(f"🖼️ Immagine caricata: {image_path}")

        model_dir = self.download_model()
        pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_dir)

        with torch.no_grad():
            mesh_textured = pipeline(mesh, image=image_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"textured_output_{timestamp}.ply")
        os.makedirs(self.output_dir, exist_ok=True)
        mesh_textured.export(output_path)
        logging.info(f"✅ Mesh texturizzata salvata in: {output_path}")
        return output_path

if __name__ == "__main__":
    tg = TextureGenerator()
    tg.verify_repo()
    tg.setup_build()
    mesh = input("Inserisci il percorso della mesh (.ply): ").strip()
    img = input("Inserisci il percorso dell'immagine da usare come texture (.png/.jpg): ").strip()
    tg.apply_texture(mesh, img)
