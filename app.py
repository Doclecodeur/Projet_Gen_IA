"""
Point d'entrée Chainlit — Interface conversationnelle.

Lancement :
    chainlit run app.py -w

Variables d'environnement requises (fichier .env) :
    OPENAI_API_KEY      — clé OpenAI (obligatoire)
    TAVILY_API_KEY      — clé Tavily pour la recherche web (recommandé)
    OPENWEATHER_API_KEY — clé OpenWeatherMap (optionnel, mode démo sinon)
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil

import chainlit as cl

from agent import AssistantRouter, RAGPipeline
from config import validate_config
from ingestion import build_vectorstore
from ingestion.document_loader import delete_document, list_indexed_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Validation des clés API au démarrage (arrêt immédiat si OPENAI_API_KEY manquante)
validate_config()

# ── Initialisation au démarrage ─────────────────────────────────────────────────


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialise le vectorstore, le pipeline RAG et le routeur au démarrage."""

    try:
        await cl.Message(
            content="⏳ Chargement des documents et initialisation de l'assistant…"
        ).send()

        vectorstore = build_vectorstore()
        rag_pipeline = RAGPipeline(vectorstore)
        router = AssistantRouter(rag_pipeline)

        # Stockage dans la session Chainlit (isolé par utilisateur)
        cl.user_session.set("router", router)
        cl.user_session.set("vectorstore", vectorstore)

        # ── Initialisation du tracking des sources RAG ──────────────────────
        cl.user_session.set("cited_sources", [])

        # Boutons d'action
        actions = [
            cl.Action(
                name="clear_memory",
                label="🗑️ Effacer la mémoire",
                description="Réinitialise l'historique de la conversation",
                value="clear",
                payload={"action": "clear_memory"},
            ),
            cl.Action(
                name="show_sources",
                label="📚 Sources citées",
                description="Affiche tous les documents RAG cités dans cette session",
                value="sources",
                payload={"action": "show_sources"},
            ),
            cl.Action(
                name="list_docs",
                label="📂 Documents indexés",
                description="Affiche les documents disponibles dans le vectorstore",
                value="list_docs",
                payload={"action": "list_docs"},
            ),
        ]

        await cl.Message(
            content=(
                "Je peux vous aider à :\n"
                "- 📄 Répondre à des questions sur vos **documents internes**\n"
                "- 🧮 Effectuer des **calculs**\n"
                "- 🌤️ Obtenir la **météo** d'une ville\n"
                "- 🔍 Rechercher des informations sur le **web**\n"
                "- ✅ Gérer votre **liste de tâches**\n\n"
                "Comment puis-je vous aider ?"
            ),
            actions=actions,
        ).send()

    except Exception as exc:
        logger.exception("Erreur lors de l'initialisation")
        await cl.Message(content=f"❌ Erreur d'initialisation : {exc}").send()


# ── Gestion des messages ────────────────────────────────────────────────────────


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Traite chaque message utilisateur, gère l'upload éventuel, puis retourne la réponse."""
    router: AssistantRouter | None = cl.user_session.get("router")

    if router is None:
        await cl.Message(
            content="❌ L'assistant n'est pas initialisé. Rechargez la page."
        ).send()
        return

    # ── Commande de suppression de document ──────────────────────────────────
    if message.content.strip().startswith("!supprimer "):
        filename = message.content.strip()[len("!supprimer ") :].strip()
        vectorstore = cl.user_session.get("vectorstore")
        if not vectorstore:
            await cl.Message(content="❌ Vectorstore non disponible.").send()
            return
        deleted = delete_document(vectorstore, filename)
        if deleted == 0:
            await cl.Message(
                content=f"⚠️ Aucun chunk trouvé pour `{filename}`. Vérifiez le nom exact avec 📂 Documents indexés."
            ).send()
        else:
            file_path = Path("documents") / filename
            if file_path.exists():
                file_path.unlink()
            await cl.Message(
                content=f"✅ `{filename}` supprimé : {deleted} chunks retirés de ChromaDB."
            ).send()
        return

    # ── Gestion de l'upload via le chat ──────────────────────────────────────
    if getattr(message, "elements", None):
        target_dir = Path("documents")
        target_dir.mkdir(parents=True, exist_ok=True)

        allowed_exts = {".pdf", ".docx", ".txt", ".json", ".xlsx", ".xls"}
        imported_files: list[str] = []

        for element in message.elements:
            file_path = getattr(element, "path", None)
            file_name = getattr(element, "name", None)

            if not file_path or not file_name:
                continue

            ext = Path(file_name).suffix.lower()
            if ext not in allowed_exts:
                await cl.Message(
                    content=(
                        f"⚠️ Format non supporté pour `{file_name}`.\n"
                        "Formats acceptés : PDF, DOCX, TXT, JSON, XLSX, XLS."
                    )
                ).send()
                continue

            source = Path(file_path)
            target = target_dir / file_name
            shutil.copy2(source, target)
            imported_files.append(file_name)

        if imported_files:
            vectorstore = build_vectorstore(force_reload=True)
            rag_pipeline = RAGPipeline(vectorstore)
            router = AssistantRouter(rag_pipeline)
            cl.user_session.set("router", router)
            cl.user_session.set(
                "vectorstore", vectorstore
            )  # sync avec le nouveau vectorstore

            await cl.Message(
                content=(
                    "✅ Fichier(s) ajouté(s) et indexé(s) : "
                    + ", ".join(f"`{name}`" for name in imported_files)
                )
            ).send()

        # Si l'utilisateur envoie seulement un fichier sans question, on s'arrête ici
        if not message.content or not message.content.strip():
            return

    # ── Traitement normal de la question ─────────────────────────────────────
    async with cl.Step(name="Analyse de la question…") as step:
        result = await router.route_async(message.content)
        step.output = f"Source utilisée : **{result['source'].upper()}**"

    badge_map = {
        "rag": "📄 *Réponse basée sur les documents internes*",
        "agent": "🤖 *Réponse de l'agent (outils ou conversation directe)*",
    }
    badge = badge_map.get(result["source"], "")
    response_text: str = result["answer"]
    final_response = f"{badge}\n\n{response_text}" if badge else response_text

    # ── Tracking des sources RAG citées ──────────────────────────────────────
    if result["source"] == "rag":
        cited = cl.user_session.get("cited_sources", [])
        for doc in result.get("sources", []):
            source_name = doc.metadata.get("source", "inconnu")
            page = doc.metadata.get("page")
            entry = {"source": source_name, "page": page}
            if entry not in cited:
                cited.append(entry)
        cl.user_session.set("cited_sources", cited)

    # Documents sources affichés en panneau latéral (uniquement pour le RAG)
    source_docs = result.get("sources", [])
    elements: list[cl.Text] = []

    for i, doc in enumerate(source_docs, start=1):
        source_name: str = doc.metadata.get("source", "Document")
        raw_page = doc.metadata.get("page")
        page_label = f" (p.{raw_page + 1})" if isinstance(raw_page, int) else ""
        label = f"Source {i} — {source_name}{page_label}"
        elements.append(
            cl.Text(
                name=label,
                content=doc.page_content.strip(),
                display="side",
            )
        )

    await cl.Message(content=final_response, elements=elements).send()


# ── Actions utilisateur ─────────────────────────────────────────────────────────


@cl.action_callback("clear_memory")
async def on_clear_memory(action: cl.Action) -> None:
    """Réinitialise la mémoire conversationnelle sur demande de l'utilisateur."""
    router: AssistantRouter | None = cl.user_session.get("router")
    if router:
        router.clear_memory()

    # Réinitialisation du tracking des sources
    cl.user_session.set("cited_sources", [])

    await cl.Message(
        content="🗑️ Mémoire effacée. Nouvelle conversation démarrée."
    ).send()
    await action.remove()


@cl.action_callback("list_docs")
async def on_list_docs(action: cl.Action) -> None:
    """Affiche la liste des documents indexés dans ChromaDB dans un step collapsable."""
    vectorstore = cl.user_session.get("vectorstore")
    if not vectorstore:
        await cl.Message(content="❌ Vectorstore non disponible.").send()
        return

    files = list_indexed_files(vectorstore)
    if not files:
        await cl.Message(content="Aucun document indexé pour l'instant.").send()
        return

    lines = [f"- `{f}`" for f in files]
    async with cl.Step(name=f"📂 {len(files)} document(s) indexé(s)") as step:
        step.output = (
            "\n".join(lines)
            + "\n\n💡 *Pour supprimer un document, tapez :* `!supprimer NomDuFichier.pdf`"
        )


@cl.action_callback("show_sources")
async def on_show_sources(action: cl.Action) -> None:
    """Affiche la liste des documents RAG cités dans la session courante."""
    cited = cl.user_session.get("cited_sources", [])
    if not cited:
        await cl.Message(content="Aucune source citée pour l'instant.").send()
    else:
        lines = [
            f"- `{e['source']}`"
            + (f" p.{e['page'] + 1}" if isinstance(e.get("page"), int) else "")
            for e in cited
        ]
        await cl.Message(
            content="📚 **Sources citées dans cette session :**\n" + "\n".join(lines)
        ).send()
