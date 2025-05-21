import os
import subprocess
import shutil
import sys
import threading
import itertools
import time
import wave
import psutil
from datetime import datetime
from pyannote.audio import Pipeline

# === CONFIGURATION ===
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
assert hf_token, "⚠️ Le token Hugging Face est manquant (HUGGINGFACE_TOKEN)."

input_folder = "input"
output_folder = "output"
archived_folder = "archived"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(archived_folder, exist_ok=True)

AUDIO_EXTENSIONS = (".m4a", ".mp3", ".wav", ".mp4", ".mov", ".mkv", ".avi")

# === SPINNER ANIMÉ POUR PATIENTER ===
def spinning_cursor(message="⏳ Traitement..."):
    spinner = itertools.cycle(['|', '/', '-', '\\'])
    while not getattr(threading.current_thread(), "stop", False):
        sys.stdout.write(f'\r{message} ' + next(spinner))
        sys.stdout.flush()
        time.sleep(0.1)

# === MESURE D’UTILISATION SYSTÈME ===
def print_system_usage():
    ram = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    disk = psutil.disk_io_counters()
    
    read_mb = disk.read_bytes // (1024 ** 2)
    write_mb = disk.write_bytes // (1024 ** 2)
    
    print(f"🧠 CPU : {cpu:.1f}%   |   RAM : {ram.percent:.1f}% ({ram.used // (1024**2)} Mo / {ram.total // (1024**2)} Mo)")
    print(f"💾 Disque : {read_mb} Mo lus  |  {write_mb} Mo écrits")

# === DURÉE DU FICHIER AUDIO ===
def get_duration(wav_file):
    with wave.open(wav_file, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return round(frames / float(rate), 2)

# === INITIALISATION DU PIPELINE PYANNOTE ===
print("\n🔁 Initialisation du modèle de diarisation (pyannote)...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
print("✅ Modèle chargé avec succès.")

# === LISTAGE DES FICHIERS À TRAITER ===
input_files = [f for f in os.listdir(input_folder) if f.lower().endswith(AUDIO_EXTENSIONS)]
if not input_files:
    print(f"\n⚠️ Aucun fichier audio ou vidéo trouvé dans '{input_folder}/'.")
    exit()

print(f"\n🎯 {len(input_files)} fichier(s) détecté(s) à traiter.\n")

# === TRAITEMENT PAR FICHIER ===
for idx, filename in enumerate(input_files, 1):
    base_name = os.path.splitext(filename)[0]
    input_path = os.path.join(input_folder, filename)
    wav_path = os.path.join(output_folder, f"{base_name}.wav")
    rttm_path = os.path.join(output_folder, f"{base_name}.rttm")

    print(f"\n================= {idx}/{len(input_files)} =================")
    print(f"🗂️  Fichier : {filename}")
    print(f"🕒 Début : {datetime.now().strftime('%H:%M:%S')}")

    # === 1. CONVERSION EN WAV ===
    print("🎧 Étape 1 - Conversion en WAV...")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        wav_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✅ Conversion terminée.")

    duration = get_duration(wav_path)
    print(f"🕓 Durée audio : {int(duration // 60)} min {int(duration % 60)} sec")
    est_time = round(duration * 1.3)
    print(f"⏱️ Estimation du temps de traitement : ~{int(est_time // 60)} min {int(est_time % 60)} sec")

    print_system_usage()

    # === 2. DIARISATION AVEC SPINNER ===
    print("🧠 Étape 2 - Diarisation en cours...")

    spinner_thread = threading.Thread(target=spinning_cursor, args=("⏳ Analyse des voix en cours...",))
    spinner_thread.start()

    start = time.time()
    diarization = pipeline(wav_path)
    end = time.time()

    spinner_thread.stop = True
    spinner_thread.join()
    sys.stdout.write("\r✅ Diarisation terminée.                        \n")

    # === 3. SAUVEGARDE DU RTTM ===
    with open(rttm_path, "w") as f:
        diarization.write_rttm(f)
    print(f"📝 Fichier RTTM enregistré : {rttm_path}")

    # === 4. ARCHIVAGE DU FICHIER ORIGINAL ===
    archived_path = os.path.join(archived_folder, filename)
    shutil.move(input_path, archived_path)
    print(f"📦 Fichier archivé : {archived_path}")

    # === 5. AFFICHAGE TEMPS FINAL + METRICS ===
    elapsed = int(end - start)
    print(f"⏳ Temps réel de traitement : {elapsed // 60} min {elapsed % 60} sec")
    print_system_usage()

print("\n🎉 Tous les fichiers ont été traités avec succès.")