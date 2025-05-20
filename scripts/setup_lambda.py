import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)

REPO_URL = "https://github.com/deepbeepmeep/Hunyuan3D-2GP.git"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Hunyuan3D-2GP"))
REQUIREMENTS_FILE = "requirements_lambda.txt"


def run_installation(cmd, use_user_flag=False):
    """Esegue un comando pip con gestione errori e flag --user se necessario."""
    try:
        if use_user_flag:
            cmd += ["--user"]
            logging.warning(f"Riprovo con flag --user: {' '.join(cmd)}")
        else:
            logging.info(f"Esecuzione: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Errore durante: {' '.join(cmd)}\n{e}")
        if not use_user_flag:
            return run_installation(cmd, use_user_flag=True)
        return False


def clone_repository():
    """Clona il repository se non già presente."""
    if not os.path.exists(REPO_DIR):
        logging.info(f"📥 Clonazione repository {REPO_URL}...")
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
        logging.info("✅ Repository clonato.")
    else:
        logging.info("✅ Repository già presente.")


def install_editable():
    """Installa il repository Hunyuan in modalità editable."""
    logging.info("🔧 Installazione in modalità editable...")
    subprocess.run(["pip", "install", "-e", ".", "--user"], cwd=REPO_DIR, check=True)
    logging.info("✅ Installazione editable completata.")


def install_pytorch():
    """Installa PyTorch e torchvision compatibili con CUDA 12.4."""
    logging.info("🔄 Installazione PyTorch con CUDA 12.4...")
    torch_cmd = [
        sys.executable, "-m", "pip", "install",
        "torch==2.5.1",
        "torchvision==0.18.1",
        "--index-url", "https://download.pytorch.org/whl/test/cu124"
    ]
    if not run_installation(torch_cmd):
        sys.exit("❌ Installazione PyTorch fallita.")


def install_requirements():
    """Installa le dipendenze da requirements_lambda.txt, creandolo se non esiste."""
    requirements_path = os.path.join(BASE_DIR, REQUIREMENTS_FILE)
    if not os.path.exists(requirements_path):
        logging.warning(f"{REQUIREMENTS_FILE} non trovato. Creo un file minimo...")
        with open(requirements_path, "w") as f:
            f.write("diffusers\nhuggingface_hub\naccelerators\ntransformers\nscipy\n")

    logging.info(f"📦 Installazione dipendenze da {REQUIREMENTS_FILE}...")
    requirements_cmd = [
        sys.executable, "-m", "pip", "install", "-r", requirements_path
    ]
    if not run_installation(requirements_cmd):
        sys.exit("❌ Installazione dipendenze fallita.")


def check_pytorch_cuda():
    """Verifica che PyTorch sia installato e che la GPU CUDA sia visibile."""
    logging.info("🔍 Verifica installazione PyTorch e disponibilità CUDA...")
    try:
        subprocess.check_call([
            sys.executable, "-c",
            (
                "import torch; "
                "print(f'✅ PyTorch {torch.__version__} | CUDA disponibile: {torch.cuda.is_available()}"
                f"{' | Versione CUDA: ' + torch.version.cuda if torch.cuda.is_available() else ''}')"
            )
        ])
    except subprocess.CalledProcessError:
        logging.warning("⚠️  Verifica PyTorch fallita.")


def full_setup():
    """Esegue il setup completo: clone repo, install editable, PyTorch e requirements."""
    clone_repository()
    install_editable()
    install_pytorch()
    install_requirements()
    check_pytorch_cuda()
    logging.info("✅ Setup completato con successo.")


if __name__ == "__main__":
    full_setup()
