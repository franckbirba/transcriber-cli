import argparse
import os
import subprocess
import tempfile
import whisper
from datetime import timedelta
from pyannote.core import Segment, Annotation
from tqdm import tqdm
import time

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def parse_rttm(rttm_file):
    from pyannote.core import Annotation
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

def transcribe_file(audio_path, rttm_path, output_path, model):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    annotation = parse_rttm(rttm_path)
    segments = list(annotation.itersegments(with_label=True))
    output_lines = []

    start_time = time.time()
    for segment, speaker in tqdm(segments, desc=f"⏳ {base_name}", unit="seg"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

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
        result = model.transcribe(tmp_path, language="fr", fp16=False)
        os.remove(tmp_path)

        start_str = format_time(segment.start)
        end_str = format_time(segment.end)
        speaker_str = speaker.capitalize()
        transcript = result['text'].strip()
        output_lines.append(f"[{start_str} - {end_str}] {speaker_str}: {transcript}")

    final_path = os.path.join(output_path, f"{base_name}.txt")
    with open(final_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    total_time = int(time.time() - start_time)
    print(f"\n✅ {base_name} terminé en {total_time // 60} min {total_time % 60} sec")
    print(f"📝 Fichier : {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Transcription audio multilocuteur avec Whisper + RTTM")
    parser.add_argument("--input", type=str, default="output", help="Répertoire contenant .wav et .rttm")
    parser.add_argument("--output", type=str, default="transcripts", help="Répertoire de sortie")
    parser.add_argument("--model", type=str, default="base", help="Modèle Whisper à utiliser")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model = whisper.load_model(args.model)
    files = [f for f in os.listdir(args.input) if f.endswith(".rttm")]

    for rttm_file in files:
        base = rttm_file.replace(".rttm", "")
        wav_file = os.path.join(args.input, base + ".wav")
        rttm_path = os.path.join(args.input, rttm_file)

        if os.path.exists(wav_file):
            transcribe_file(wav_file, rttm_path, args.output, model)
        else:
            print(f"❌ Fichier WAV manquant pour {base}")

if __name__ == "__main__":
    main()