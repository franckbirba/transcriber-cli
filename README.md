# Local Transcriber

[![PyPI version](https://badge.fury.io/py/local-transcriber.svg)](https://badge.fury.io/py/local-transcriber)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Local Transcriber** est un outil puissant et automatisé pour la transcription audio multilocuteur, basé sur [Whisper](https://github.com/openai/whisper) et [Pyannote.audio](https://github.com/pyannote/pyannote-audio). Il permet de transformer des fichiers audio en transcriptions textuelles tout en identifiant les différents locuteurs.

---

## Fonctionnalités

- **Diarisation des locuteurs** : Identification des segments audio par locuteur.
- **Transcription précise** : Utilisation des modèles Whisper pour une transcription de haute qualité.
- **Support multilingue** : Compatible avec plusieurs langues.
- **Automatisation complète** : Gestion des étapes de diarisation et de transcription en une seule commande.
- **Installation simplifiée** : Détection et installation automatique des dépendances nécessaires (comme `ffmpeg`).

---

## Installation

### Prérequis

- Python 3.8 ou supérieur
- `ffmpeg` installé et accessible dans le `PATH`
- Un token Hugging Face pour utiliser les modèles de diarisation ([Créer un token ici](https://huggingface.co/settings/tokens)).

### Installation via pip

```bash
pip install local-transcriber