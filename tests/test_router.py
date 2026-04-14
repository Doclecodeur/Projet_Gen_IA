"""
Tests unitaires pour le routeur décisionnel.

Vérifie que le routage RAG / Agent est correct selon le score de similarité,
que la mémoire est bien mise à jour, et que les erreurs sont gérées.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from agent.router import AssistantRouter


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_rag_pipeline(sample_documents):
    """Pipeline RAG mocké."""
    pipeline = MagicMock()
    pipeline.answer.return_value = ("Réponse RAG avec citation.", sample_documents)
    pipeline.answer_async = AsyncMock(return_value=("Réponse RAG async.", sample_documents))
    pipeline.retrieve_with_scores.return_value = [
        (doc, 0.85) for doc in sample_documents
    ]
    pipeline.vectorstore = MagicMock()
    return pipeline


@pytest.fixture
def mock_rag_pipeline_no_docs(mock_rag_pipeline):
    """Pipeline RAG sans documents pertinents (score faible)."""
    mock_rag_pipeline.retrieve_with_scores.return_value = [
        (MagicMock(), 0.15)  # Score sous le seuil de 0.4
    ]
    return mock_rag_pipeline


@pytest.fixture
def router(mock_rag_pipeline):
    """Routeur avec agent mocké."""
    with patch("agent.router.ChatOpenAI"), \
         patch("agent.router.create_openai_tools_agent"), \
         patch("agent.router.AgentExecutor"), \
         patch("agent.router.RunnableWithMessageHistory") as mock_rwmh, \
         patch("agent.router.get_all_tools", return_value=[]):

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Réponse agent."}
        mock_agent.ainvoke = AsyncMock(return_value={"output": "Réponse agent async."})
        mock_rwmh.return_value = mock_agent

        r = AssistantRouter(mock_rag_pipeline)
        r._agent_executor = mock_agent
        return r


@pytest.fixture
def router_no_docs(mock_rag_pipeline_no_docs):
    """Routeur avec score RAG insuffisant → doit router vers l'agent."""
    with patch("agent.router.ChatOpenAI"), \
         patch("agent.router.create_openai_tools_agent"), \
         patch("agent.router.AgentExecutor"), \
         patch("agent.router.RunnableWithMessageHistory") as mock_rwmh, \
         patch("agent.router.get_all_tools", return_value=[]):

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Réponse agent."}
        mock_agent.ainvoke = AsyncMock(return_value={"output": "Réponse agent async."})
        mock_rwmh.return_value = mock_agent

        r = AssistantRouter(mock_rag_pipeline_no_docs)
        r._agent_executor = mock_agent
        return r


# ══════════════════════════════════════════════════════════════════════════════
# Détection de documents pertinents
# ══════════════════════════════════════════════════════════════════════════════

class TestHasRelevantDocuments:
    """Tests du mécanisme de seuillage par score de similarité."""

    def test_score_eleve_retourne_true(self, router):
        assert router._has_relevant_documents("Politique de congés") is True

    def test_score_faible_retourne_false(self, router_no_docs):
        assert router_no_docs._has_relevant_documents("Météo à Paris") is False

    def test_score_exactement_au_seuil(self, mock_rag_pipeline):
        """Score exactement égal au seuil → doit retourner True."""
        mock_rag_pipeline.retrieve_with_scores.return_value = [
            (MagicMock(), 0.4)  # Exactement le seuil
        ]
        with patch("agent.router.ChatOpenAI"), \
             patch("agent.router.create_openai_tools_agent"), \
             patch("agent.router.AgentExecutor"), \
             patch("agent.router.RunnableWithMessageHistory"), \
             patch("agent.router.get_all_tools", return_value=[]):
            r = AssistantRouter(mock_rag_pipeline)
        assert r._has_relevant_documents("Question limite") is True

    def test_exception_retourne_false(self, router):
        """En cas d'erreur du vectorstore, ne doit pas lever d'exception."""
        router._rag.retrieve_with_scores.side_effect = RuntimeError("ChromaDB down")
        result = router._has_relevant_documents("Question")
        assert result is False


# ══════════════════════════════════════════════════════════════════════════════
# Routage synchrone
# ══════════════════════════════════════════════════════════════════════════════

class TestRoute:
    """Tests du routage synchrone."""

    def test_route_vers_rag_si_score_eleve(self, router):
        result = router.route("Quelle est la politique de congés ?")
        assert result["source"] == "rag"
        assert len(result["sources"]) > 0
        assert isinstance(result["answer"], str)

    def test_route_vers_agent_si_score_faible(self, router_no_docs):
        result = router_no_docs.route("Quelle est la météo à Paris ?")
        assert result["source"] == "agent"
        assert result["sources"] == []

    def test_memoire_mise_a_jour_apres_rag(self, router):
        router.route("Politique de congés ?")
        history = router.chat_history
        assert len(history) == 2  # 1 message user + 1 message AI

    def test_result_contient_toutes_les_cles(self, router):
        result = router.route("Une question")
        assert "answer" in result
        assert "source" in result
        assert "sources" in result

    def test_erreur_agent_retourne_message_propre(self, router_no_docs):
        """Une erreur de l'agent ne doit jamais faire crasher l'application."""
        router_no_docs._agent_executor.invoke.side_effect = RuntimeError("API error")
        result = router_no_docs.route("Question impossible")
        assert result["source"] == "agent"
        assert "erreur" in result["answer"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# Routage asynchrone
# ══════════════════════════════════════════════════════════════════════════════

class TestRouteAsync:
    """Tests du routage asynchrone."""

    @pytest.mark.asyncio
    async def test_route_async_vers_rag(self, router):
        result = await router.route_async("Politique de congés ?")
        assert result["source"] == "rag"

    @pytest.mark.asyncio
    async def test_route_async_vers_agent(self, router_no_docs):
        result = await router_no_docs.route_async("Météo à Lyon ?")
        assert result["source"] == "agent"

    @pytest.mark.asyncio
    async def test_erreur_agent_async_geree(self, router_no_docs):
        router_no_docs._agent_executor.ainvoke.side_effect = RuntimeError("Async error")
        result = await router_no_docs.route_async("Question problématique")
        assert "erreur" in result["answer"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# Gestion de la mémoire
# ══════════════════════════════════════════════════════════════════════════════

class TestMemory:
    """Tests de la gestion de la mémoire conversationnelle."""

    def test_clear_memory_vide_historique(self, router):
        router.route("Question 1")
        assert len(router.chat_history) > 0
        router.clear_memory()
        assert len(router.chat_history) == 0

    def test_trim_history_limite_la_taille(self, router):
        """La fenêtre glissante ne doit pas dépasser MEMORY_WINDOW_SIZE * 2 messages."""
        from config import MEMORY_WINDOW_SIZE
        # Simule 15 tours de conversation (au-delà de la fenêtre de 10)
        for i in range(15):
            router._history.add_user_message(f"Question {i}")
            router._history.add_ai_message(f"Réponse {i}")
        router._trim_history()
        assert len(router.chat_history) <= MEMORY_WINDOW_SIZE * 2
