# 🤖 Assistant intelligent RAG + Agents

Assistant conversationnel combinant **RAG documentaire** et **agents à outils**, développé avec **LangChain**, **OpenAI**, **Chainlit** et **ChromaDB**.

L’application choisit automatiquement entre :
- une **réponse fondée sur vos documents** via un pipeline RAG,
- ou une **réponse outillée** via un agent capable d’utiliser plusieurs outils.

---

## Aperçu

Cet assistant permet de :

- interroger un corpus de documents internes ;
- obtenir des réponses avec **citations de sources** ;
- utiliser des outils externes ou utilitaires ;
- conserver une **mémoire conversationnelle courte** ;
- basculer automatiquement entre **RAG** et **agent conversationnel** selon la question posée.

---

## Fonctionnalités

### 📄 RAG documentaire
- indexation de documents locaux ;
- recherche sémantique avec **ChromaDB** ;
- réponses contextualisées à partir du corpus ;
- citations de type : `[Source, p.X]`.

### 🤖 Agent à outils
L’agent peut utiliser plusieurs outils selon le besoin :
- **calculatrice** ;
- **météo** ;
- **recherche web** ;
- **lecture d’une todo list** ;
- **ajout d’un élément dans la todo list** ;
- **réponse conversationnelle directe** si aucun outil n’est nécessaire.

### 🧠 Mémoire conversationnelle
- conservation des **10 derniers tours** ;
- amélioration de la continuité des échanges ;
- possibilité de réinitialiser l’historique.

---

## Architecture

```text
Utilisateur
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      AssistantRouter                        │
│                                                             │
│   Question → score de similarité ChromaDB ≥ 0.4 ?          │
│                      │                                      │
│              ┌───────┴────────┐                             │
│            Oui               Non                            │
│              │                │                             │
│              ▼                ▼                             │
│        RAGPipeline      AgentExecutor                       │
│        (ChromaDB        (OpenAI Tools Agent)                │
│         + Citations)          │                             │
│                         ┌─────┴──────────────────┐         │
│                         │  Outils disponibles     │         │
│                         │  ├─ Calculatrice        │         │
│                         │  ├─ Météo               │         │
│                         │  ├─ Recherche web       │         │
│                         │  ├─ Lecture todo        │         │
│                         │  └─ Ajout todo          │         │
│                         └────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
 ChatMessageHistory (fenêtre glissante, 10 derniers tours)