import subprocess
import sys
import argparse
from dotenv import load_dotenv
import os

# Charger les variables d'environnement depuis .env
load_dotenv()
default_input = os.getenv("INPUT_FOLDER", "input")
default_output = os.getenv("OUTPUT_FOLDER", "output")
default_model = os.getenv("WHISPER_MODEL", "base")

# === PARSEUR D'ARGUMENTS CLI ===
parser = argparse.ArgumentParser(description="Pipeline complet pour la diarisation et la transcription audio")
parser.add_argument("--gpu", action="store_true", help="Force l'utilisation du GPU pour Pyannote et Whisper")
parser.add_argument("--lang", type=str, default="fr", help="Langue de transcription (ex: 'fr', 'en', 'es')")
parser.add_argument("--model", type=str, default=default_model, help="Modèle Whisper à utiliser (1=tiny, 2=base, 3=small, 4=medium, 5=large, 6=large-v3)")
parser.add_argument("--input", type=str, default=default_input, help="Dossier contenant les fichiers audio à traiter")
parser.add_argument("--output", type=str, default=default_output, help="Dossier où enregistrer les fichiers traités")
args = parser.parse_args()

# === LANCEMENT DIARISATION ===
print("\n========== ÉTAPE 1 : DIARISATION ==========")
try:
    diarize_command = [
        "python", "diarize.py",
        "--input", args.input,
        "--output", args.output
    ]
    if args.gpu:
        diarize_command.append("--gpu")
    subprocess.run(diarize_command, check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ Erreur pendant la diarisation : {e}")
    sys.exit(1)

# === LANCEMENT TRANSCRIPTION ===
print("\n========== ÉTAPE 2 : TRANSCRIPTION ==========")
try:
    transcribe_command = [
        "python", "transcribe_segments.py",
        "--model", args.model,
        "--lang", args.lang,
        "--output", args.output
    ]
    if args.gpu:
        transcribe_command.append("--gpu")
    subprocess.run(transcribe_command, check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ Erreur pendant la transcription : {e}")
    sys.exit(1)

print("\n✅ Processus complet terminé avec succès.")