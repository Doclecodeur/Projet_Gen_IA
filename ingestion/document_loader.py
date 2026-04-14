from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)


def _enrich_metadata(source: Path, extra: dict | None = None) -> dict:
    metadata = {
        "source": source.name,
        "file_name": source.name,
        "extension": source.suffix.lower(),
    }
    if extra:
        metadata.update(extra)
    return metadata

def _load_pdf(path: Path) -> list[Document]:
    docs = PyPDFLoader(str(path)).load()
    enriched_docs: list[Document] = []

    for doc in docs:
        meta = dict(doc.metadata) if doc.metadata else {}
        meta.update(_enrich_metadata(path))
        enriched_docs.append(Document(page_content=doc.page_content, metadata=meta))

    return enriched_docs


def _load_docx(path: Path) -> list[Document]:
    docs = Docx2txtLoader(str(path)).load()
    enriched_docs: list[Document] = []

    for doc in docs:
        meta = dict(doc.metadata) if doc.metadata else {}
        meta.update(_enrich_metadata(path))
        enriched_docs.append(Document(page_content=doc.page_content, metadata=meta))

    return enriched_docs


def _load_txt(path: Path) -> list[Document]:
    content = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not content:
        return []

    return [
        Document(
            page_content=content,
            metadata=_enrich_metadata(path),
        )
    ]


def _load_json(path: Path) -> list[Document]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    docs: list[Document] = []

    if isinstance(data, dict):
        docs.append(
            Document(
                page_content=json.dumps(data, ensure_ascii=False, indent=2),
                metadata=_enrich_metadata(path, {"json_type": "object"}),
            )
        )
        return docs

    if isinstance(data, list):
        if not data:
            return []

        for idx, item in enumerate(data, start=1):
            docs.append(
                Document(
                    page_content=json.dumps(item, ensure_ascii=False, indent=2),
                    metadata=_enrich_metadata(
                        path,
                        {
                            "json_type": "list_item",
                            "item_index": idx,
                        },
                    ),
                )
            )
        return docs

    docs.append(
        Document(
            page_content=str(data),
            metadata=_enrich_metadata(path, {"json_type": "scalar"}),
        )
    )
    return docs


def _load_excel(path: Path) -> list[Document]:
    docs: list[Document] = []

    excel_file = pd.ExcelFile(path)
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet_name).fillna("")

        if df.empty:
            continue

        for row_idx, row in df.iterrows():
            row_content = []
            for col in df.columns:
                value = str(row[col]).strip()
                if value:
                    row_content.append(f"{col}: {value}")

            if not row_content:
                continue

            docs.append(
                Document(
                    page_content=(
                        f"Feuille: {sheet_name}\n"
                        f"Ligne Excel: {row_idx + 2}\n"
                        + "\n".join(row_content)
                    ),
                    metadata=_enrich_metadata(
                        path,
                        {
                            "sheet_name": sheet_name,
                            "row_number": int(row_idx + 2),
                        },
                    ),
                )
            )

    return docs


_LOADER_MAP: dict[str, Callable[[Path], list[Document]]] = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".txt": _load_txt,
    ".json": _load_json,
    ".xlsx": _load_excel,
    ".xls": _load_excel,
}


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """Charge tous les documents supportés depuis docs_dir."""
    if not docs_dir.exists():
        logger.warning("Le dossier de documents n'existe pas : %s", docs_dir)
        return []

    all_docs: list[Document] = []

    files = sorted(
        path
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _LOADER_MAP
    )

    if not files:
        logger.warning("Aucun fichier supporté trouvé dans %s", docs_dir)
        return []

    for file_path in files:
        ext = file_path.suffix.lower()
        loader = _LOADER_MAP[ext]

        try:
            docs = loader(file_path)
            logger.info("Chargé %d document(s) depuis %s", len(docs), file_path.name)
            all_docs.extend(docs)
        except Exception as exc:
            logger.warning("Erreur lors du chargement de %s : %s", file_path, exc)

    if not all_docs:
        logger.warning("Aucun contenu exploitable n'a été chargé depuis %s", docs_dir)

    return all_docs


def split_documents(documents: list[Document]) -> list[Document]:
    """Découpe les documents en chunks adaptés à l'indexation vectorielle."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Découpage : %d chunks produits à partir de %d documents",
        len(chunks),
        len(documents),
    )
    return chunks


def _make_embeddings() -> OpenAIEmbeddings:
    """Instancie le modèle d'embedding OpenAI."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )


def build_vectorstore(force_reload: bool = False) -> Chroma:
    """
    Crée ou charge la base vectorielle ChromaDB.
    """
    embeddings = _make_embeddings()

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    existing_docs = vectorstore.get()
    existing_count = len(existing_docs.get("ids", []))

    if existing_count > 0 and not force_reload:
        logger.info(
            "Collection existante chargée (%d vecteurs). "
            "Utilisez force_reload=True pour re-indexer.",
            existing_count,
        )
        return vectorstore

    logger.info("Indexation des documents en cours…")
    documents = load_documents()

    if not documents:
        logger.warning(
            "Aucun document à indexer. Déposez des fichiers supportés dans '%s'.",
            DOCS_DIR,
        )
        return vectorstore

    chunks = split_documents(documents)

    if force_reload and existing_count > 0:
        logger.info("Suppression de la collection existante pour re-indexation.")
        vectorstore.delete_collection()
        vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    vectorstore.add_documents(chunks)
    logger.info("Indexation terminée : %d chunks stockés.", len(chunks))
    return vectorstore


def get_document_count(vectorstore: Chroma) -> int:
    """Retourne le nombre de chunks indexés dans la collection."""
    result = vectorstore.get()
    return len(result.get("ids", []))

def delete_document(vectorstore: Chroma, filename: str) -> int:
    """
    Supprime tous les chunks d'un document spécifique depuis ChromaDB.

    Parameters
    ----------
    vectorstore : Chroma
        Instance ChromaDB active.
    filename : str
        Nom du fichier à supprimer (ex: "Resume_Antibiotiques.docx").

    Returns
    -------
    int
        Nombre de chunks supprimés.
    """
    results = vectorstore.get(include=["metadatas"])
    ids_to_delete = [
        doc_id
        for doc_id, meta in zip(results["ids"], results["metadatas"])
        if meta.get("file_name") == filename or meta.get("source") == filename
    ]

    if not ids_to_delete:
        logger.warning("Aucun chunk trouvé pour le fichier : %s", filename)
        return 0

    vectorstore.delete(ids=ids_to_delete)
    logger.info("%d chunks supprimés pour : %s", len(ids_to_delete), filename)
    return len(ids_to_delete)


def list_indexed_files(vectorstore: Chroma) -> list[str]:
    """
    Retourne la liste des fichiers uniques indexés dans ChromaDB.

    Returns
    -------
    list[str]
        Noms des fichiers indexés, triés alphabétiquement.
    """
    results = vectorstore.get(include=["metadatas"])
    files = sorted({
        meta.get("file_name") or meta.get("source", "inconnu")
        for meta in results["metadatas"]
    })
    return files
