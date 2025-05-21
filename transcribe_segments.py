import os
from dotenv import load_dotenv
import argparse
import whisper
import subprocess
import tempfile
from datetime import timedelta
from pyannote.core import Segment, Annotation
from pathlib import Path
import sys
from tqdm import tqdm
import time

# Charger les variables d'environnement depuis .env
load_dotenv()
default_output = os.getenv("OUTPUT_FOLDER", "output")
default_model = os.getenv("WHISPER_MODEL", "base")

# === PARSEUR D'ARGUMENTS CLI ===
parser = argparse.ArgumentParser(description="Transcription audio multilocuteur avec Whisper")
parser.add_argument("--model", type=str, default=default_model, help="Modèle Whisper à utiliser (1=tiny, 2=base, 3=small, 4=medium, 5=large, 6=large-v3)")
parser.add_argument("--lang", type=str, default="fr", help="Langue de transcription (ex: 'fr', 'en', 'es')")
parser.add_argument("--gpu", action="store_true", help="Force l'utilisation du GPU pour Whisper")
parser.add_argument("--output", type=str, default=default_output, help="Dossier où enregistrer les fichiers traités")
args = parser.parse_args()

# === CONFIG ===
OUTPUT_DIR = args.output  # Utilisation de l'argument CLI ou de la valeur par défaut depuis .env
TRANSCRIPTS_DIR = "transcripts"
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# === CHOIX DU MODÈLE WHISPER ===
model_map = {
    "1": "tiny", "2": "base", "3": "small",
    "4": "medium", "5": "large", "6": "large-v3"
}
model_name = model_map.get(args.model, default_model)

print(f"\n🧠 Chargement du modèle Whisper '{model_name}'...")
device = "cuda" if args.gpu else "cpu"
model = whisper.load_model(model_name, device=device)
print(f"✅ Modèle prêt sur {device.upper()}.")

# === LANGUE DE TRANSCRIPTION ===
language = args.lang
print(f"🌍 Langue sélectionnée : {language}")

# === UTILS ===
def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def parse_rttm(rttm_file):
    annotation = Annotation()
    with open(rttm_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segment = Segment(start, start + duration)
            annotation[segment] = speaker
    return annotation

# === LISTAGE DES FICHIERS RTTM ===
files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".rttm")]

# Si aucun fichier RTTM n'est trouvé dans OUTPUT_DIR, vérifier dans TRANSCRIPTS_DIR
if not files:
    print(f"⚠️ Aucun fichier RTTM trouvé dans '{OUTPUT_DIR}'. Vérification dans '{TRANSCRIPTS_DIR}'...")
    files = [f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".rttm")]

# Si toujours aucun fichier, quitter proprement
if not files:
    print("❌ Aucun fichier à transcrire. Assurez-vous d'avoir des fichiers RTTM prêts à être transcrits.")
    exit()

print(f"🎯 {len(files)} fichier(s) RTTM détecté(s) à transcrire.\n")

# === TRAITEMENT DE TOUS LES FICHIERS ===
for rttm_file in files:
    base_name = rttm_file.replace(".rttm", "")
    audio_path = os.path.join(OUTPUT_DIR, base_name + ".wav")
    rttm_path = os.path.join(OUTPUT_DIR, rttm_file)

    if not os.path.exists(audio_path):
        print(f"❌ Audio manquant pour {rttm_file}. Ignoré.")
        continue

    print(f"\n🎙️  Transcription de : {base_name}")

    annotation = parse_rttm(rttm_path)
    # Remplacez l'appel à itersegments avec l'argument `with_label`
    segments = [(segment, annotation[segment]) for segment in annotation.itersegments()]
    output_lines = []

    start_time = time.time()

    for segment, speaker in tqdm(segments, desc="   ⏳ Transcription", unit="seg"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        # Extraction audio du segment
        subprocess.run([
            "ffmpeg", "-y",
            "-i", audio_path,
            "-ss", str(segment.start),
            "-to", str(segment.end),
            "-ar", "16000",
            "-ac", "1",
            "-loglevel", "error",
            tmp_path
        ])

        # Transcription Whisper
        result = model.transcribe(tmp_path, language=language, fp16=False)
        os.remove(tmp_path)

        # Formatage
        start_str = format_time(segment.start)
        end_str = format_time(segment.end)
        speaker_str = speaker.capitalize()
        transcript = result['text'].strip()

        line = f"[{start_str} - {end_str}] {speaker_str}: {transcript}"
        output_lines.append(line)

    # Sauvegarde finale
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{base_name}.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    total_time = int(time.time() - start_time)
    print(f"\n✅ Transcription terminée : {transcript_path}")
    print(f"🕒 Durée : {total_time // 60} min {total_time % 60} sec | {len(segments)} segments")

print("\n🎉 Tous les fichiers ont été transcrits avec succès.")