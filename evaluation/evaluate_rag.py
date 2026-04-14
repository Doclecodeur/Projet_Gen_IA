"""
Évaluation quantitative du pipeline RAG.

Métriques calculées :
  - Precision@K    : proportion de chunks retrouvés pertinents parmi les K retournés
  - Recall@K       : proportion de chunks pertinents retrouvés sur ceux attendus
  - MRR            : Mean Reciprocal Rank (position du premier résultat pertinent)
  - Answer Accuracy : % de réponses contenant les mots-clés attendus
  - Latence moyenne : temps de réponse end-to-end

Usage :
    python evaluation/evaluate_rag.py

Prérequis :
    - Les documents de test doivent être indexés dans ChromaDB
    - OPENAI_API_KEY doit être définie dans .env

Documents indexés (2164 chunks) :
    - Biostatistique_et_analyse_informatique_des_données_de_R.pdf  (608 chunks)
    - dataanalysisforthelifesciences.pdf                           (1012 chunks)
    - Méthodes-statistiques-pour-l'analyse-de-données-de-comptage.pdf (501 chunks)
    - Devoir Python.pdf                                            (40 chunks)
    - Resume_Antibiotiques.docx                                    (3 chunks)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

from agent.rag_pipeline import RAGPipeline
from config import validate_config
from ingestion import build_vectorstore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Jeu de données d'évaluation ────────────────────────────────────────────────
# Dataset construit à partir des documents réellement indexés dans ChromaDB.
# Couvre les 5 sources : biostatistique R, data analysis life sciences,
# données de comptage surdispersées, devoir Python (scraping Pokémon), antibiotiques.

EVALUATION_DATASET = [
    # ── Biostatistique avec R ──────────────────────────────────────────────────
    {
        "id": "Q1",
        "question": "À quoi servent les biostatistiques médicales selon Lalanne et Mesbah ?",
        "relevant_keywords": ["biostatistique", "praticien", "clinique", "santé"],
        "expected_answer_keywords": ["biostatistique", "praticien", "clinique"],
    },
    {
        "id": "Q2",
        "question": "Quelles étapes couvre le livre de biostatistique avec R, de l'importation à la modélisation ?",
        "relevant_keywords": ["importation", "modélisation", "données", "statistique"],
        "expected_answer_keywords": ["importation", "modélisation", "R"],
    },
    # ── Data Analysis for the Life Sciences ───────────────────────────────────
    {
        "id": "Q3",
        "question": "Qu'est-ce qu'un permutation test et dans quel contexte est-il utilisé ?",
        "relevant_keywords": ["permutation", "test", "association"],
        "expected_answer_keywords": ["permutation", "test"],
    },
    {
        "id": "Q4",
        "question": "Comment réaliser une analyse exploratoire avec des boxplots et des scatterplots ?",
        "relevant_keywords": ["boxplot", "scatterplot", "exploratory", "correlation"],
        "expected_answer_keywords": ["boxplot", "scatterplot"],
    },
    # ── Données de comptage surdispersées ─────────────────────────────────────
    {
        "id": "Q5",
        "question": "Quelles méthodes statistiques sont adaptées pour analyser des données de comptage surdispersées ?",
        "relevant_keywords": ["comptage", "surdispersées", "méthodes", "statistiques"],
        "expected_answer_keywords": ["comptage", "surdispersées"],
    },
    # ── Devoir Python / Web scraping Pokémon ──────────────────────────────────
    {
        "id": "Q6",
        "question": "Comment scraper les statistiques d'un Pokémon depuis pokemondb.net avec BeautifulSoup ?",
        "relevant_keywords": ["BeautifulSoup", "scraping", "pokémon", "vitals-table"],
        "expected_answer_keywords": ["BeautifulSoup", "pokémon", "table"],
    },
    {
        "id": "Q7",
        "question": "Quelles sont les statistiques de base de Bulbasaur (HP, Attack, Speed) ?",
        "relevant_keywords": ["Bulbasaur", "HP", "Attack", "Speed", "stats"],
        "expected_answer_keywords": ["Bulbasaur", "45", "49"],
    },
    # ── Antibiotiques ──────────────────────────────────────────────────────────
    {
        "id": "Q8",
        "question": "Quel est le mécanisme d'action des bêta-lactamines ?",
        "relevant_keywords": ["bêta-lactamine", "paroi", "transpeptidase", "bactéricide"],
        "expected_answer_keywords": ["bêta-lactamine", "paroi", "bactéricide"],
    },
    {
        "id": "Q9",
        "question": "Quelle est la différence entre un antibiotique bactéricide et bactériostatique ?",
        "relevant_keywords": ["bactéricide", "bactériostatique", "antibiotique"],
        "expected_answer_keywords": ["bactéricide", "bactériostatique"],
    },
    {
        "id": "Q10",
        "question": "Quel est le mécanisme d'action des tétracyclines ?",
        "relevant_keywords": ["tétracycline", "ribosome", "30S", "protéines"],
        "expected_answer_keywords": ["tétracycline", "30S", "ribosome"],
    },
]


# ── Structures de données ──────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    """Résultat de l'évaluation pour une question."""
    question_id: str
    question: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    answer_contains_expected: bool
    latency_seconds: float
    retrieved_sources: List[str] = field(default_factory=list)
    answer_preview: str = ""


@dataclass
class EvaluationReport:
    """Rapport global d'évaluation."""
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    answer_accuracy: float          # % de réponses contenant les mots-clés attendus
    mean_latency_seconds: float
    num_questions: int
    k: int = 4                      # Valeur de K utilisée pour Precision@K et Recall@K
    results: List[QuestionResult] = field(default_factory=list)


# ── Métriques de retrieval ─────────────────────────────────────────────────────

def compute_precision_at_k(
    retrieved_contents: List[str],
    relevant_keywords: List[str],
    k: int,
) -> float:
    """
    Precision@K : parmi les K chunks retournés, combien contiennent
    au moins un mot-clé pertinent ?
    """
    if not retrieved_contents:
        return 0.0
    top_k = retrieved_contents[:k]
    relevant_count = sum(
        1 for content in top_k
        if any(kw.lower() in content.lower() for kw in relevant_keywords)
    )
    return relevant_count / len(top_k)


def compute_recall_at_k(
    retrieved_contents: List[str],
    relevant_keywords: List[str],
    k: int,
) -> float:
    """
    Recall@K : parmi les mots-clés attendus, combien sont couverts
    par les K chunks retournés ?
    """
    if not relevant_keywords or not retrieved_contents:
        return 0.0
    top_k_text = " ".join(retrieved_contents[:k]).lower()
    covered = sum(1 for kw in relevant_keywords if kw.lower() in top_k_text)
    return covered / len(relevant_keywords)


def compute_reciprocal_rank(
    retrieved_contents: List[str],
    relevant_keywords: List[str],
) -> float:
    """
    Reciprocal Rank : inverse du rang du premier résultat pertinent.
    RR = 1 si le premier chunk est pertinent, 0.5 si le second, etc.
    """
    for rank, content in enumerate(retrieved_contents, start=1):
        if any(kw.lower() in content.lower() for kw in relevant_keywords):
            return 1.0 / rank
    return 0.0


def check_answer_quality(
    answer: str,
    expected_keywords: List[str],
) -> bool:
    """Vérifie que la réponse contient les mots-clés attendus."""
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords)


# ── Évaluation principale ──────────────────────────────────────────────────────

def evaluate_pipeline(
    rag_pipeline: RAGPipeline,
    dataset: list[dict],
    k: int = 4,
) -> EvaluationReport:
    """
    Évalue le pipeline RAG sur le jeu de données fourni.

    Parameters
    ----------
    rag_pipeline : RAGPipeline
        Pipeline à évaluer.
    dataset : list[dict]
        Questions avec mots-clés de référence.
    k : int
        Nombre de chunks à considérer (Precision@K, Recall@K).

    Returns
    -------
    EvaluationReport
        Rapport avec toutes les métriques agrégées.
    """
    results: List[QuestionResult] = []

    for item in dataset:
        logger.info("Évaluation de %s : %s", item["id"], item["question"])

        start_time = time.perf_counter()
        answer, source_docs = rag_pipeline.answer(item["question"])
        latency = time.perf_counter() - start_time

        retrieved_contents = [doc.page_content for doc in source_docs]
        # Normalise le chemin Windows/Linux pour n'afficher que le nom du fichier
        retrieved_sources = [
            doc.metadata.get("source", "?").replace("\\", "/").split("/")[-1]
            for doc in source_docs
        ]

        precision = compute_precision_at_k(
            retrieved_contents, item["relevant_keywords"], k
        )
        recall = compute_recall_at_k(
            retrieved_contents, item["relevant_keywords"], k
        )
        rr = compute_reciprocal_rank(retrieved_contents, item["relevant_keywords"])
        answer_ok = check_answer_quality(answer, item["expected_answer_keywords"])

        results.append(
            QuestionResult(
                question_id=item["id"],
                question=item["question"],
                precision_at_k=round(precision, 3),
                recall_at_k=round(recall, 3),
                reciprocal_rank=round(rr, 3),
                answer_contains_expected=answer_ok,
                latency_seconds=round(latency, 3),
                retrieved_sources=retrieved_sources,
                answer_preview=answer[:150] + "…" if len(answer) > 150 else answer,
            )
        )

    # ── Agrégation ─────────────────────────────────────────────────────────────
    n = len(results)
    report = EvaluationReport(
        mean_precision_at_k=round(sum(r.precision_at_k for r in results) / n, 3),
        mean_recall_at_k=round(sum(r.recall_at_k for r in results) / n, 3),
        mean_reciprocal_rank=round(sum(r.reciprocal_rank for r in results) / n, 3),
        answer_accuracy=round(sum(r.answer_contains_expected for r in results) / n, 3),
        mean_latency_seconds=round(sum(r.latency_seconds for r in results) / n, 3),
        num_questions=n,
        k=k,
        results=results,
    )
    return report


def print_report(report: EvaluationReport) -> None:
    """Affiche le rapport d'évaluation dans le terminal."""
    print("\n" + "═" * 60)
    print("       RAPPORT D'ÉVALUATION — PIPELINE RAG")
    print("═" * 60)
    print(f"  Nombre de questions     : {report.num_questions}")
    print(f"  Precision@{report.k:<14}   : {report.mean_precision_at_k:.1%}")
    print(f"  Recall@{report.k:<17}   : {report.mean_recall_at_k:.1%}")
    print(f"  Mean Reciprocal Rank    : {report.mean_reciprocal_rank:.3f}")
    print(f"  Answer Accuracy         : {report.answer_accuracy:.1%}")
    print(f"  Latence moyenne         : {report.mean_latency_seconds:.2f}s")
    print("═" * 60)
    print("\n  Détail par question :\n")
    for r in report.results:
        status = "✅" if r.answer_contains_expected else "❌"
        print(f"  {status} [{r.question_id}] {r.question[:55]}…")
        print(f"     Precision: {r.precision_at_k:.2f} | Recall: {r.recall_at_k:.2f} "
              f"| RR: {r.reciprocal_rank:.2f} | {r.latency_seconds:.2f}s")
        print(f"     Sources : {', '.join(r.retrieved_sources) or 'aucune'}")
        print()
    print("═" * 60)


def save_report(report: EvaluationReport, output_path: Path) -> None:
    """Sauvegarde le rapport en JSON pour traçabilité."""
    data = asdict(report)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Rapport sauvegardé : %s", output_path)


# ── Point d'entrée ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    validate_config()

    logger.info("Chargement du vectorstore…")
    vectorstore = build_vectorstore()
    pipeline = RAGPipeline(vectorstore)

    logger.info("Évaluation sur %d questions…", len(EVALUATION_DATASET))
    report = evaluate_pipeline(pipeline, EVALUATION_DATASET, k=4)

    print_report(report)

    output_file = Path("evaluation") / "rapport_evaluation.json"
    save_report(report, output_file)
