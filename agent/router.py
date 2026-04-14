"""
Routeur décisionnel : RAG / Agent / Conversation directe.

Logique de décision :
  1. Si des documents pertinents sont trouvés (score de similarité suffisant)
     → Pipeline RAG répond avec citations.
  2. Sinon, l'agent LangChain tente de répondre via ses outils.
  3. L'agent dispose d'un accès à l'historique de conversation pour maintenir
     la cohérence des échanges.

La mémoire conversationnelle est gérée via `ChatMessageHistory` (API stable
LangChain ≥ 0.2, remplaçant `ConversationBufferWindowMemory` dépréciée).
"""

from __future__ import annotations

import logging
from typing import List, TypedDict

try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
except ImportError:
    from langchain_core.agents import AgentExecutor
    from langchain.agents import create_openai_tools_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from agent.rag_pipeline import RAGPipeline
from agent.tools import get_all_tools
from config import (
    AGENT_SYSTEM_PROMPT,
    AGENT_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    MEMORY_WINDOW_SIZE,
    OPENAI_API_KEY,
    RAG_SIMILARITY_THRESHOLD,
    RETRIEVAL_TOP_K,
)

logger = logging.getLogger(__name__)


# ── Type de retour structuré ───────────────────────────────────────────────────

class RouterResult(TypedDict):
    """Résultat standardisé retourné par le routeur."""
    answer: str        # Texte de la réponse
    source: str        # "rag" | "agent"
    sources: list      # Documents sources (si RAG), sinon liste vide


# ── Routeur principal ──────────────────────────────────────────────────────────

class AssistantRouter:
    """
    Orchestre le routage entre RAG et agent à outils.

    Parameters
    ----------
    rag_pipeline:
        Instance de ``RAGPipeline`` déjà initialisée.
    """

    def __init__(self, rag_pipeline: RAGPipeline) -> None:
        self._rag = rag_pipeline

        # Historique de conversation partagé entre le RAG et l'agent
        self._history = ChatMessageHistory()

        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=AGENT_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            openai_api_key=OPENAI_API_KEY,
        )

        self._agent_executor = self._build_agent()

    # ── Construction de l'agent ────────────────────────────────────────────────

    def _build_agent(self) -> RunnableWithMessageHistory:
        """
        Construit l'agent avec ses outils et l'encapsule dans
        `RunnableWithMessageHistory` pour la gestion de la mémoire.
        """
        tools = get_all_tools()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", AGENT_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_openai_tools_agent(
            llm=self._llm,
            tools=tools,
            prompt=prompt,
        )

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,          # Passer à True pour déboguer les appels d'outils
            handle_parsing_errors=True,
            max_iterations=5,       # Évite les boucles infinies
            return_intermediate_steps=False,
        )

        # Encapsulation avec gestion de l'historique (API stable LangChain ≥ 0.2)
        return RunnableWithMessageHistory(
            executor,
            lambda session_id: self._history,  # session unique par instance
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    # ── Logique de routage ─────────────────────────────────────────────────────

    def _has_relevant_documents(self, query: str) -> bool:
        """
        Vérifie si des documents suffisamment pertinents existent pour la requête.

        Utilise ``RAGPipeline.retrieve_with_scores`` (API publique) pour éviter
        tout accès aux attributs privés de ChromaDB.
        """
        try:
            results = self._rag.retrieve_with_scores(query, k=RETRIEVAL_TOP_K)
            if not results:
                return False
            best_score = max(score for _, score in results)
            logger.debug("Score de similarité maximal : %.3f (seuil : %.3f)", best_score, RAG_SIMILARITY_THRESHOLD)
            return best_score >= RAG_SIMILARITY_THRESHOLD
        except Exception as exc:  # noqa: BLE001
            logger.warning("Erreur lors du calcul de similarité : %s — fallback vers l'agent.", exc)
            return False

    def _trim_history(self) -> None:
        """Conserve uniquement les N derniers tours de conversation en mémoire."""
        messages = self._history.messages
        max_messages = MEMORY_WINDOW_SIZE * 2  # 1 tour = 1 msg user + 1 msg AI
        if len(messages) > max_messages:
            self._history.messages = messages[-max_messages:]

    # ── Points d'entrée publics ────────────────────────────────────────────────

    def route(self, user_message: str) -> RouterResult:
        """
        Route le message utilisateur vers la bonne source de réponse.

        Returns
        -------
        RouterResult
            Dictionnaire avec les clés ``answer``, ``source`` et ``sources``.
        """
        if self._has_relevant_documents(user_message):
            logger.info("Routage → RAG")
            answer, source_docs = self._rag.answer(user_message)
            # Synchronise manuellement l'historique pour les réponses RAG
            self._history.add_user_message(user_message)
            self._history.add_ai_message(answer)
            self._trim_history()
            return RouterResult(answer=answer, source="rag", sources=source_docs)

        logger.info("Routage → Agent")
        try:
            result = self._agent_executor.invoke(
                {"input": user_message},
                config={"configurable": {"session_id": "default"}},
            )
            self._trim_history()
            return RouterResult(answer=result.get("output", ""), source="agent", sources=[])
        except Exception as exc:  # noqa: BLE001
            logger.error("Erreur agent : %s", exc)
            return RouterResult(
                answer=f"Une erreur est survenue lors du traitement : {exc}",
                source="agent",
                sources=[],
            )

    async def route_async(self, user_message: str) -> RouterResult:
        """Version asynchrone du routeur pour Chainlit."""
        if self._has_relevant_documents(user_message):
            logger.info("Routage async → RAG")
            answer, source_docs = await self._rag.answer_async(user_message)
            self._history.add_user_message(user_message)
            self._history.add_ai_message(answer)
            self._trim_history()
            return RouterResult(answer=answer, source="rag", sources=source_docs)

        logger.info("Routage async → Agent")
        try:
            result = await self._agent_executor.ainvoke(
                {"input": user_message},
                config={"configurable": {"session_id": "default"}},
            )
            self._trim_history()
            return RouterResult(answer=result.get("output", ""), source="agent", sources=[])
        except Exception as exc:  # noqa: BLE001
            logger.error("Erreur agent async : %s", exc)
            return RouterResult(
                answer=f"Une erreur est survenue : {exc}",
                source="agent",
                sources=[],
            )

    @property
    def chat_history(self) -> List[BaseMessage]:
        """Retourne l'historique de conversation en cours."""
        return self._history.messages

    def clear_memory(self) -> None:
        """Réinitialise la mémoire conversationnelle."""
        self._history.clear()
        logger.info("Mémoire conversationnelle réinitialisée.")


