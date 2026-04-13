"""
Configuration centralisée du projet RAG Agent.

Toutes les constantes et paramètres sont regroupés ici.
La fonction `validate_config()` doit être appelée au démarrage de l'application
afin de détecter les clés manquantes avant tout appel API.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# load_dotenv() doit être appelé avant toute lecture de os.getenv()
load_dotenv()

# ── Chemins ────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent
DOCS_DIR: Path = BASE_DIR / "documents"  # Dossier contenant les fichiers à ingérer
CHROMA_DIR: Path = BASE_DIR / "chroma_db"  # Persistance de la base vectorielle

DOCS_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# ── Clés API ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")

# ── LLM — modèle commun ────────────────────────────────────────────────────────
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_MAX_TOKENS: int = 2048

# Température 0 pour le RAG (réponses factuelles, déterministes)
RAG_TEMPERATURE: float = 0.0
# Température légèrement plus haute pour l'agent (formulation plus naturelle)
AGENT_TEMPERATURE: float = 0.2

# ── Embeddings ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "text-embedding-3-small"

# ── Vectorstore ────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = "documents_internes"
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 100
RETRIEVAL_TOP_K: int = 4  # Nombre de chunks retournés par le retriever
RAG_SIMILARITY_THRESHOLD: float = 0.4  # Score minimum pour considérer un doc pertinent

# ── Mémoire conversationnelle ──────────────────────────────────────────────────
MEMORY_WINDOW_SIZE: int = 10  # Nombre de tours de conversation conservés

# ── Outils externes ────────────────────────────────────────────────────────────
OPENWEATHER_BASE_URL: str = "https://api.openweathermap.org/data/2.5/weather"
WEB_SEARCH_MAX_RESULTS: int = 3

# ── Prompts système ────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT: str = """\
Tu es un assistant expert qui répond en t'appuyant uniquement sur les extraits \
de documents fournis ci-dessous. Cite toujours la source (nom du fichier et \
numéro de page si disponible) entre crochets, par exemple [manuel_rh.pdf, p.3]. \
Si les documents ne contiennent pas la réponse, dis-le clairement plutôt que \
d'inventer une information.

Extraits pertinents :
{context}\
"""

AGENT_SYSTEM_PROMPT: str = """\
Tu es un assistant polyvalent disposant d'outils spécialisés.
Réponds en français, de façon concise et précise.

Règles importantes :
- Pour toute question sur l'heure actuelle, utilise l'outil dédié.
- Pour toute question sur la date actuelle, utilise l'outil dédié si disponible.
- Pour la météo, utilise l'outil météo uniquement si une ville précise est donnée.
- N'invente jamais l'heure, la date ou la météo.
"""


# ── Validation ─────────────────────────────────────────────────────────────────


def validate_config() -> None:
    """
    Vérifie que les variables d'environnement obligatoires sont définies.

    Lève une `EnvironmentError` explicite si OPENAI_API_KEY est absente,
    et affiche des avertissements pour les clés optionnelles manquantes.
    """
    if not OPENAI_API_KEY:
        sys.exit(
            "OPENAI_API_KEY est manquante.\n"
            "   Créez un fichier .env à la racine du projet et ajoutez :\n"
            "   OPENAI_API_KEY=sk-..."
        )

    warnings: list[str] = []
    if not TAVILY_API_KEY:
        warnings.append("TAVILY_API_KEY absente — la recherche web sera désactivée.")
    if not OPENWEATHER_API_KEY:
        warnings.append(
            "OPENWEATHER_API_KEY absente — la météo fonctionnera en mode démo."
        )

    for warning in warnings:
        print(f"{warning}")
