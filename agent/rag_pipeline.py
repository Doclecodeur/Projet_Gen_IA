"""
Pipeline RAG (Retrieval-Augmented Generation).

Responsabilités :
- Récupérer les chunks les plus pertinents depuis ChromaDB.
- Construire le contexte documentaire avec citations.
- Générer la réponse finale via le LLM.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from config import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    OPENAI_API_KEY,
    RAG_SYSTEM_PROMPT,
    RAG_TEMPERATURE,
    RETRIEVAL_TOP_K,
)

logger = logging.getLogger(__name__)


def _format_docs_with_citations(docs: List[Document]) -> str:
    """
    Formate les documents récupérés en bloc de contexte avec citations.

    Exemple de sortie :
        [Source : manuel_rh.pdf, p.3]
        Le congé annuel est de 25 jours ouvrés…
    """
    formatted: List[str] = []
    for doc in docs:
        source: str = doc.metadata.get("source", "source inconnue")
        # PyPDFLoader indexe les pages à partir de 0 → affichage à partir de 1
        raw_page = doc.metadata.get("page")
        page_label = f", p.{raw_page + 1}" if isinstance(raw_page, int) else ""
        citation = f"[Source : {source}{page_label}]"
        formatted.append(f"{citation}\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)


class RAGPipeline:
    """
    Pipeline RAG encapsulant retriever + LLM.

    Parameters
    ----------
    vectorstore:
        Instance ``Chroma`` déjà initialisée et peuplée.
    """

    def __init__(self, vectorstore: Chroma) -> None:
        # Stocké publiquement pour permettre au routeur de calculer
        # les scores de similarité sans accès aux attributs privés.
        self.vectorstore: Chroma = vectorstore

        self._retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_TOP_K},
        )

        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=RAG_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            openai_api_key=OPENAI_API_KEY,
        )

        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )

        # Chaîne LangChain Expression Language (LCEL)
        self._chain = (
            {
                "context": self._retriever | _format_docs_with_citations,
                "question": RunnablePassthrough(),
            }
            | self._prompt
            | self._llm
            | StrOutputParser()
        )

    def retrieve(self, query: str) -> List[Document]:
        """Retourne les documents pertinents sans générer de réponse."""
        return self._retriever.invoke(query)

    def retrieve_with_scores(self, query: str, k: int = RETRIEVAL_TOP_K) -> List[Tuple[Document, float]]:
        """
        Retourne les documents avec leur score de similarité cosinus.

        Utilisé par le routeur pour décider si le RAG est pertinent.
        """
        return self.vectorstore.similarity_search_with_relevance_scores(query, k=k)

    def answer(self, question: str) -> Tuple[str, List[Document]]:
        """
        Génère une réponse RAG avec citations.

        Returns
        -------
        tuple[str, list[Document]]
            Texte de la réponse et liste des documents sources utilisés.
        """
        source_docs = self.retrieve(question)

        if not source_docs:
            logger.info("Aucun document pertinent trouvé pour : « %s »", question)
            return (
                "Je n'ai pas trouvé d'information pertinente dans les documents "
                "disponibles pour répondre à cette question.",
                [],
            )

        response: str = self._chain.invoke(question)
        logger.debug("Réponse RAG générée (%d chars)", len(response))
        return response, source_docs

    async def answer_async(self, question: str) -> Tuple[str, List[Document]]:
        """Version asynchrone de ``answer`` pour Chainlit (streaming natif)."""
        source_docs = self.retrieve(question)

        if not source_docs:
            return (
                "Je n'ai pas trouvé d'information pertinente dans les documents.",
                [],
            )

        response: str = await self._chain.ainvoke(question)
        return response, source_docs
