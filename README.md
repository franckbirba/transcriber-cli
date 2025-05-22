# Local Transcriber

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/franckbirba/transcriber-cli/blob/main/transcriber_colab.ipynb)

[![PyPI version](https://badge.fury.io/py/local-transcriber.svg)](https://badge.fury.io/py/local-transcriber)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Local Transcriber** est un outil puissant et automatisé pour la transcription audio et vidéo multilocuteur, basé sur [Whisper](https://github.com/openai/whisper) et [Pyannote.audio](https://github.com/pyannote/pyannote-audio). Il permet de transformer des fichiers audio ou vidéo en transcriptions textuelles tout en identifiant les différents locuteurs.

---

## Fonctionnalités

- **Diarisation des locuteurs** : Identification des segments audio par locuteur.
- **Transcription précise** : Utilisation des modèles Whisper pour une transcription de haute qualité.
- **Support multilingue** : Compatible avec plusieurs langues.
- **Prise en charge des formats audio et vidéo** : `.wav`, `.mp3`, `.mp4`, `.MOV`, etc.
- **Automatisation complète** : Gestion des étapes de diarisation et de transcription en une seule commande.
- **Organisation des fichiers** : Les fichiers traités sont archivés automatiquement.

---

## Installation

### Prérequis

- Python 3.8 ou supérieur
- `ffmpeg` installé et accessible dans le `PATH`
- Un token Hugging Face pour utiliser les modèles de diarisation ([Créer un token ici](https://huggingface.co/settings/tokens)).

### Installation via pip

```bash
pip install local-transcriber
```

---

## Structure des dossiers

Lors de l'exécution, le script utilise la structure suivante :

```
input/         # Contient les fichiers audio ou vidéo à traiter
output/        # Contient les fichiers générés (RTTM, WAV, etc.)
├── transcripts/  # Contient les transcriptions finales au format .txt
├── archived/     # Contient les fichiers originaux après traitement
```

---

## Utilisation

### Lancer dans Google Colab

Cliquez sur le bouton ci-dessous pour exécuter le projet directement dans Google Colab :

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/franckbirba/transcriber-cli/blob/main/transcriber_colab.ipynb)

### Utilisation en local

1. Placez vos fichiers audio ou vidéo dans le dossier `input`.
2. Exécutez la commande suivante :

```bash
python run_all.py --input input --output output --gpu
```

#### Options disponibles :
- `--input` : Dossier contenant les fichiers à traiter (par défaut : `input`).
- `--output` : Dossier où enregistrer les fichiers générés (par défaut : `output`).
- `--gpu` : Utilise le GPU si disponible pour accélérer le traitement.

---

## Dépendances

Les principales dépendances sont :
- [PyTorch](https://pytorch.org/) : Pour l'exécution des modèles.
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio) : Pour la diarisation des locuteurs.
- [Whisper](https://github.com/openai/whisper) : Pour la transcription audio.
- `ffmpeg` : Pour la conversion des fichiers audio/vidéo.

Installez toutes les dépendances avec :

```bash
pip install -r requirements.txt
```

---

## Licence

Ce projet est sous licence [MIT](https://opensource.org/licenses/MIT).