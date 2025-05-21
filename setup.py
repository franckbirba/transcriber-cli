import os
import subprocess
import sys
import shutil
from setuptools import setup, find_packages

REQUIRED_DIRS = ["input", "output", "archived", "transcripts"]

def print_header():
    print("=" * 50)
    print("🛠️  INSTALLATION AJUSTÉE POUR TRANSCRIPTION LOCALE")
    print("=" * 50)

def install_missing_packages():
    try:
        with open("requirements.txt") as f:
            required_packages = f.read().splitlines()
        for package in required_packages:
            try:
                __import__(package.split("-")[0])
            except ImportError:
                print(f"📦 Installation de {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", package])
    except FileNotFoundError:
        print("❌ Fichier requirements.txt introuvable.")
        sys.exit(1)

def check_ffmpeg():
    print("\n🔍 Vérification de ffmpeg...")
    if shutil.which("ffmpeg") is None:
        print("❌ ffmpeg n'est pas installé ou pas dans le PATH.")
        print("➡️  Installation automatique de ffmpeg...")

        if sys.platform == "darwin":  # macOS
            subprocess.run(["brew", "install", "ffmpeg"], check=True)
        elif sys.platform.startswith("linux"):  # Linux
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y", "ffmpeg"], check=True)
        else:
            print("⚠️ Installation automatique non prise en charge pour ce système.")
            print("Veuillez installer ffmpeg manuellement.")
            sys.exit(1)

        if shutil.which("ffmpeg") is None:
            print("❌ L'installation de ffmpeg a échoué. Veuillez l'installer manuellement.")
            sys.exit(1)
        else:
            print("✅ ffmpeg installé avec succès.")
    else:
        print("✅ ffmpeg détecté.")

def ensure_env_token():
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        print("✅ Token Hugging Face déjà présent dans l'environnement.")
        return

    print("\n🔐 Ton token Hugging Face est requis pour pyannote.audio")
    print("👉 Crée-le ici : https://huggingface.co/settings/tokens")
    token = input("Colle ici ton token (commence par hf_...): ").strip()

    if not token.startswith("hf_"):
        print("❌ Token invalide.")
        sys.exit(1)

    with open(".env", "w", encoding="utf-8") as f:
        f.write(f"HUGGINGFACE_TOKEN={token}\n")

    print("✅ Token enregistré dans .env")
    print("⚠️ Pense à exécuter `make export-env` pour charger les variables d'environnement.")

def create_directories():
    for d in REQUIRED_DIRS:
        os.makedirs(d, exist_ok=True)
    print("✅ Dossiers créés : input/, output/, archived/, transcripts/")

# Lecture des dépendances depuis requirements.txt
try:
    with open("requirements.txt") as f:
        install_requires = f.read().splitlines()
except FileNotFoundError:
    print("❌ Fichier requirements.txt introuvable.")
    install_requires = []

setup(
    name="local-transcriber",
    version="0.1",
    packages=find_packages(),  # ou [""] si pas de packages déclarés
    py_modules=["transcriber_cli"],
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "transcriber = transcriber_cli:main"
        ]
    },
)    

if __name__ == "__main__":
    print_header()
    install_missing_packages()
    check_ffmpeg()
    ensure_env_token()
    create_directories()
    print("\n✅ Installation terminée. Tu peux maintenant lancer :")
    print("   → `transcriber` pour lancer le service installé.")
    print("      Arguments disponibles :")
    print("         --input <fichier> : Spécifie le fichier audio à transcrire.")
    print("         --output <fichier> : Spécifie le fichier de sortie pour la transcription.")
    print("         --lang <code> : Définit la langue de transcription (ex: 'fr', 'en').")
    print("         --help : Affiche l'aide pour les options disponibles.")