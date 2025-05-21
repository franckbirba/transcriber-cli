# Makefile pour transcripteur local

# === VARS ===
PYTHON=python3
SETUP_SCRIPT=setup.py
RUN_SCRIPT=run_all.py
ENV_FILE=.env

# === COMMANDES ===

setup:
	@echo "⚙️  Installation interactive..."
	$(PYTHON) $(SETUP_SCRIPT)

export-env:
	@echo "📤 Export des variables d'environnement..."
	@export $$(cat $(ENV_FILE) | xargs)

run:
	@echo "🚀 Lancement complet (diarisation + transcription)..."
	@$(PYTHON) $(RUN_SCRIPT)

reset:
	@echo "🧹 Nettoyage des dossiers output/, transcripts/, archived/..."
	@rm -rf output/*.wav output/*.rttm transcripts/*.txt archived/*
	@echo "✅ Nettoyage terminé."

help:
	@echo "Commandes disponibles :"
	@echo "  make setup        → installation interactive"
	@echo "  make export-env   → export des variables d’environnement (.env)"
	@echo "  make run          → exécution complète (diarisation + transcription)"
	@echo "  make reset        → nettoyage des fichiers générés"