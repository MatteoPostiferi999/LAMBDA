import os
import subprocess
import sys

# Configurazione
REPO_URL = "https://github.com/deepbeepmeep/Hunyuan3D-2GP.git"
REPO_DIR = "Hunyuan3D-2GP"
REQUIREMENTS_FILE = "requirements_lambda.txt"

def run_installation(cmd, use_user_flag=False):
    """Esegue un comando di installazione con gestione errori e opzione --user"""
    try:
        if use_user_flag:
            cmd = cmd + ["--user"]
            print(f"⚠️ Riprovando con flag --user: {' '.join(cmd)}")
        else:
            print(f"🔄 Esecuzione: {' '.join(cmd)}")
        
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore durante '{' '.join(cmd)}': {e}")
        if not use_user_flag:
            print("Riproverò con il flag --user...")
            return run_installation(cmd, use_user_flag=True)
        return False

# 1. Clona il repository se non esiste
if not os.path.exists(REPO_DIR):
    print(f"📥 Clonazione repository {REPO_URL}...")
    try:
        subprocess.check_call(["git", "clone", REPO_URL])
        print("✅ Repository clonato con successo")
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore durante il clone del repository: {e}")
        sys.exit(1)

# 2. Vai nella cartella del progetto
print(f"📂 Cambio directory a {REPO_DIR}")
os.chdir(REPO_DIR)

# 3. Installa torch e torchvision compatibili con CUDA 12.4
print("🔄 Installazione PyTorch con supporto CUDA 12.4...")
torch_cmd = [
    sys.executable, "-m", "pip", "install",
    "torch==2.5.1",
    "torchvision==0.18.1",
    "--index-url", "https://download.pytorch.org/whl/test/cu124"
]

if not run_installation(torch_cmd):
    print("❌ Installazione PyTorch fallita sia con che senza --user")
    sys.exit(1)

# 4. Installa tutte le altre dipendenze dal file custom
print(f"🔄 Installazione dipendenze da {REQUIREMENTS_FILE}...")
if not os.path.exists(REQUIREMENTS_FILE):
    print(f"⚠️ File {REQUIREMENTS_FILE} non trovato, creazione file minimo...")
    with open(REQUIREMENTS_FILE, "w") as f:
        f.write("diffusers\nhuggingface_hub\naccelerators\ntransformers\nscipy\n")

requirements_cmd = [
    sys.executable, "-m", "pip", "install",
    "-r", REQUIREMENTS_FILE
]

if not run_installation(requirements_cmd):
    print(f"❌ Installazione dipendenze da {REQUIREMENTS_FILE} fallita")
    sys.exit(1)

# 5. Verifica installazione
print("🔍 Verifica installazione PyTorch con CUDA...")
try:
    check_cmd = [
        sys.executable, "-c", 
        "import torch; print(f'PyTorch {torch.__version__}, CUDA disponible: {torch.cuda.is_available()}, ' + (f'Versione CUDA: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA non disponibile'))"
    ]
    subprocess.check_call(check_cmd)
except subprocess.CalledProcessError:
    print("⚠️ Impossibile verificare l'installazione di PyTorch")

# 6. Fine
print("✅ Setup completato con successo.")