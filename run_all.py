import subprocess
import sys

# === LANCEMENT DIARISATION ===
print("\n========== ÉTAPE 1 : DIARISATION ==========")
try:
    subprocess.run(["python", "diarize.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ Erreur pendant la diarisation : {e}")
    sys.exit(1)

# === CHOIX DU MODÈLE WHISPER ===
print("\n========== ÉTAPE 2 : TRANSCRIPTION ==========")
print("Quel modèle Whisper veux-tu utiliser ?")
print("1 - tiny\n2 - base\n3 - small\n4 - medium\n5 - large-v2\n6 - large-v3 (si installé)")
choice = input("Choix [1-6] (défaut: 2 = base) : ").strip() or "2"

# === LANCEMENT TRANSCRIPTION AVEC ARGUMENT
try:
    subprocess.run(["python", "transcribe_segments.py", choice], check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ Erreur pendant la transcription : {e}")
    sys.exit(1)

print("\n✅ Processus complet terminé avec succès.")