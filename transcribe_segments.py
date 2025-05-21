import os
import subprocess
import tempfile
import whisper
from datetime import timedelta
from pyannote.core import Segment, Annotation
from pathlib import Path
import sys
from tqdm import tqdm
import time
import argparse  # Ajout pour gérer les arguments CLI

# === CONFIG ===
OUTPUT_DIR = "output"
TRANSCRIPTS_DIR = "transcripts"
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# === PARSEUR D'ARGUMENTS CLI ===
parser = argparse.ArgumentParser(description="Transcription audio multilocuteur avec Whisper")
parser.add_argument("--model", type=str, default="2", help="Modèle Whisper à utiliser (1=tiny, 2=base, 3=small, 4=medium, 5=large, 6=large-v3)")
parser.add_argument("--lang", type=str, default="fr", help="Langue de transcription (ex: 'fr', 'en', 'es')")
args = parser.parse_args()

# === CHOIX DU MODÈLE WHISPER ===
model_map = {
    "1": "tiny", "2": "base", "3": "small",
    "4": "medium", "5": "large", "6": "large-v3"
}
model_name = model_map.get(args.model, "base")

print(f"\n🧠 Chargement du modèle Whisper '{model_name}'...")
model = whisper.load_model(model_name)
print("✅ Modèle prêt.")

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

# === TRAITEMENT DE TOUS LES FICHIERS ===
files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".rttm")]
if not files:
    print("⚠️ Aucun fichier RTTM trouvé dans output/.")
    exit()

for rttm_file in files:
    base_name = rttm_file.replace(".rttm", "")
    audio_path = os.path.join(OUTPUT_DIR, base_name + ".wav")
    rttm_path = os.path.join(OUTPUT_DIR, rttm_file)

    if not os.path.exists(audio_path):
        print(f"❌ Audio manquant pour {rttm_file}. Ignoré.")
        continue

    print(f"\n🎙️  Transcription de : {base_name}")

    annotation = parse_rttm(rttm_path)
    segments = list(annotation.itersegments(with_label=True))
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