"""
Fixtures partagées entre tous les fichiers de tests.

Utilise des mocks pour éviter tout appel réel aux API OpenAI ou ChromaDB
pendant l'exécution des tests.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ── Variables d'environnement factices pour les tests ──────────────────────────
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Injecte des clés API factices pour éviter les EnvironmentError."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-tests")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-fake")
    monkeypatch.setenv("OPENWEATHER_API_KEY", "owm-test-fake")


# ── Documents de test ──────────────────────────────────────────────────────────
@pytest.fixture
def sample_documents() -> list[Document]:
    """Retourne une liste de documents factices représentant un corpus."""
    return [
        Document(
            page_content="Le congé annuel est de 25 jours ouvrés par an.",
            metadata={"source": "politique_rh.pdf", "page": 2},
        ),
        Document(
            page_content="En cas d'incident de sécurité, contacter immédiatement le RSSI.",
            metadata={"source": "manuel_securite.pdf", "page": 5},
        ),
        Document(
            page_content="La prime de fin d'année est versée en décembre.",
            metadata={"source": "politique_rh.pdf", "page": 8},
        ),
    ]


@pytest.fixture
def single_document() -> Document:
    """Un seul document avec métadonnées complètes."""
    return Document(
        page_content="Les heures supplémentaires sont compensées au taux de 125%.",
        metadata={"source": "contrat_type.pdf", "page": 0},
    )
