"""
Tests unitaires pour RAGPipeline (agent/rag_pipeline.py).

Couvre :
- retrieve() : retourne des documents
- retrieve_with_scores() : retourne des tuples (doc, score)
- answer() : réponse avec sources
- answer() : vectorstore vide → message clair sans exception
- answer_async() : version asynchrone
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from agent.rag_pipeline import RAGPipeline


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_vectorstore(sample_documents):
    """Vectorstore ChromaDB mocké."""
    vs = MagicMock()
    vs.as_retriever.return_value = MagicMock(
        invoke=MagicMock(return_value=sample_documents)
    )
    vs.similarity_search_with_relevance_scores.return_value = [
        (doc, 0.85) for doc in sample_documents
    ]
    return vs


@pytest.fixture
def mock_vectorstore_vide():
    """Vectorstore ChromaDB vide — aucun document."""
    vs = MagicMock()
    vs.as_retriever.return_value = MagicMock(
        invoke=MagicMock(return_value=[])
    )
    vs.similarity_search_with_relevance_scores.return_value = []
    return vs


@pytest.fixture
def rag_pipeline(mock_vectorstore):
    """RAGPipeline avec LLM et chaîne LCEL mockés."""
    with patch("agent.rag_pipeline.ChatOpenAI"):
        pipeline = RAGPipeline(mock_vectorstore)
        pipeline._chain = MagicMock()
        pipeline._chain.invoke.return_value = "Réponse RAG générée."
        pipeline._chain.ainvoke = AsyncMock(return_value="Réponse RAG async générée.")
        return pipeline


@pytest.fixture
def rag_pipeline_vide(mock_vectorstore_vide):
    """RAGPipeline avec vectorstore vide."""
    with patch("agent.rag_pipeline.ChatOpenAI"):
        pipeline = RAGPipeline(mock_vectorstore_vide)
        pipeline._chain = MagicMock()
        pipeline._chain.invoke.return_value = "Réponse."
        pipeline._chain.ainvoke = AsyncMock(return_value="Réponse.")
        return pipeline


# ══════════════════════════════════════════════════════════════════════════════
# retrieve
# ══════════════════════════════════════════════════════════════════════════════

class TestRetrieve:

    def test_retourne_liste_de_documents(self, rag_pipeline, sample_documents):
        result = rag_pipeline.retrieve("Politique de congés")
        assert isinstance(result, list)
        assert len(result) == len(sample_documents)

    def test_retourne_liste_vide_si_vectorstore_vide(self, rag_pipeline_vide):
        result = rag_pipeline_vide.retrieve("Question sans résultat")
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# retrieve_with_scores
# ══════════════════════════════════════════════════════════════════════════════

class TestRetrieveWithScores:

    def test_retourne_tuples_doc_score(self, rag_pipeline):
        results = rag_pipeline.retrieve_with_scores("Politique de congés")
        assert isinstance(results, list)
        assert all(isinstance(score, float) for _, score in results)

    def test_scores_entre_0_et_1(self, rag_pipeline):
        results = rag_pipeline.retrieve_with_scores("Question")
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_vectorstore_vide_retourne_liste_vide(self, rag_pipeline_vide):
        results = rag_pipeline_vide.retrieve_with_scores("Question")
        assert results == []


# ══════════════════════════════════════════════════════════════════════════════
# answer
# ══════════════════════════════════════════════════════════════════════════════

class TestAnswer:

    def test_retourne_reponse_et_sources(self, rag_pipeline, sample_documents):
        answer, sources = rag_pipeline.answer("Politique de congés ?")
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(sources, list)
        assert len(sources) == len(sample_documents)

    def test_vectorstore_vide_retourne_message_clair(self, rag_pipeline_vide):
        """Sans documents, doit retourner un message explicite, pas lever d'exception."""
        answer, sources = rag_pipeline_vide.answer("Question sans résultat")
        assert "pertinente" in answer.lower() or "trouvé" in answer.lower()
        assert sources == []

    def test_llm_non_appele_si_pas_de_documents(self, rag_pipeline_vide):
        """Le LLM ne doit pas être appelé si le retriever ne retourne rien."""
        rag_pipeline_vide.answer("Question")
        rag_pipeline_vide._chain.invoke.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# answer_async
# ══════════════════════════════════════════════════════════════════════════════

class TestAnswerAsync:

    @pytest.mark.asyncio
    async def test_retourne_reponse_et_sources(self, rag_pipeline, sample_documents):
        answer, sources = await rag_pipeline.answer_async("Politique de congés ?")
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert len(sources) == len(sample_documents)

    @pytest.mark.asyncio
    async def test_vectorstore_vide_retourne_message_clair(self, rag_pipeline_vide):
        answer, sources = await rag_pipeline_vide.answer_async("Question sans résultat")
        assert isinstance(answer, str)
        assert sources == []

    @pytest.mark.asyncio
    async def test_llm_non_appele_si_pas_de_documents(self, rag_pipeline_vide):
        await rag_pipeline_vide.answer_async("Question")
        rag_pipeline_vide._chain.ainvoke.assert_not_called()
