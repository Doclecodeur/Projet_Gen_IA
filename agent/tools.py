"""
Définition des outils mis à disposition de l'agent.

Chaque outil est une fonction Python décorée avec @tool (LangChain).
L'agent choisit automatiquement quel outil appeler selon le contexte.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from langchain_core.tools import tool

from config import (
    OPENWEATHER_API_KEY,
    OPENWEATHER_BASE_URL,
    TAVILY_API_KEY,
    WEB_SEARCH_MAX_RESULTS,
)

logger = logging.getLogger(__name__)


@tool
def get_current_date() -> str:
    """Retourne la date actuelle en France (Europe/Paris)."""
    try:
        now = datetime.now(ZoneInfo("Europe/Paris"))
    except ZoneInfoNotFoundError:
        now = datetime.now()
    return now.strftime("Nous sommes le %d %B %Y.")


@tool
def get_current_time() -> str:
    """Retourne l'heure actuelle à Paris."""
    try:
        now = datetime.now(ZoneInfo("Europe/Paris"))
    except ZoneInfoNotFoundError:
        now = datetime.now()
    return now.strftime("Il est actuellement %H:%M à Paris.")


@tool
def calculator(expression: str) -> str:
    """
    Évalue une expression mathématique textuelle et retourne le résultat.

    Opérations supportées : +, -, *, /, **, sqrt(), log(), sin(), cos().
    Exemple : "sqrt(144) + 3 * (10 - 4)"
    """
    safe_namespace: dict[str, object] = {
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_namespace)
        return f"Résultat : {result}"
    except ZeroDivisionError:
        return "Erreur : division par zéro."
    except Exception as exc:
        logger.debug("Erreur calculatrice pour '%s' : %s", expression, exc)
        return f"Expression invalide : {exc}"


@tool
def get_weather(city: str) -> str:
    """
    Retourne la météo actuelle pour une ville donnée.
    Le paramètre doit être une ville précise, pas un pays.
    """
    blocked_inputs = {
        "france", "belgique", "espagne", "italie", "allemagne",
        "europe", "afrique", "asie", "amérique"
    }

    if city.strip().lower() in blocked_inputs:
        return (
            "Merci d'indiquer une ville précise, par exemple Paris, Lyon ou Marseille, "
            "car je ne peux pas donner une météo unique pour tout un pays."
        )

    if not OPENWEATHER_API_KEY:
        logger.warning("OPENWEATHER_API_KEY non définie, utilisation des données démo.")
        return (
            f"[DÉMO] Météo à {city} : 18°C, partiellement nuageux, "
            "humidité 65 %, vent 12 km/h."
        )

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": "fr",
    }

    try:
        response = requests.get(OPENWEATHER_BASE_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"].capitalize()
        wind_speed = data["wind"]["speed"] * 3.6
        resolved_city = data.get("name", city)

        return (
            f"Météo à {resolved_city} : {description}, {temp:.1f}°C "
            f"(ressenti {feels_like:.1f}°C), humidité {humidity}%, "
            f"vent {wind_speed:.0f} km/h."
        )
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return f"Ville introuvable : '{city}'."
        return f"Erreur API météo : {exc}"
    except requests.exceptions.RequestException as exc:
        return f"Impossible de contacter le service météo : {exc}"


def _build_web_search_tool():
    """
    Instancie l'outil de recherche web Tavily si la clé est disponible,
    sinon retourne un outil de remplacement.
    """
    if TAVILY_API_KEY:
        from langchain_community.tools.tavily_search import TavilySearchResults

        return TavilySearchResults(
            max_results=WEB_SEARCH_MAX_RESULTS,
            name="web_search",
            description=(
                "Recherche des informations récentes sur le web. "
                "À utiliser pour les questions qui nécessitent des données actuelles "
                "ou absentes des documents internes."
            ),
        )

    @tool
    def web_search_unavailable(query: str) -> str:
        """Recherche web indisponible si TAVILY_API_KEY n'est pas configurée."""
        return (
            "La recherche web n'est pas disponible. "
            "Configurez TAVILY_API_KEY dans le fichier .env pour activer cette fonctionnalité."
        )

    return web_search_unavailable


_TODO_FILE = "todo.txt"


@tool
def read_todo_list() -> str:
    """Lit et retourne le contenu de la liste de tâches locale (todo.txt)."""
    try:
        with open(_TODO_FILE, encoding="utf-8") as f:
            content = f.read().strip()
        return content if content else "La liste de tâches est vide."
    except FileNotFoundError:
        return "Aucune liste de tâches trouvée (fichier todo.txt absent)."


@tool
def add_todo_item(item: str) -> str:
    """
    Ajoute une tâche à la liste locale (todo.txt).
    """
    try:
        with open(_TODO_FILE, "a", encoding="utf-8") as f:
            f.write(f"- {item}\n")
        return f"Tâche ajoutée : '{item}'."
    except OSError as exc:
        return f"Impossible d'écrire dans le fichier todo.txt : {exc}"


def get_all_tools() -> list:
    """Retourne la liste complète des outils disponibles pour l'agent."""
    return [
        calculator,
        get_current_date,
        get_current_time,
        get_weather,
        _build_web_search_tool(),
        read_todo_list,
        add_todo_item,
    ]
