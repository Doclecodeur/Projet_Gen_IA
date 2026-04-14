def calculator_tool(expression):
    try:
        return str(eval(expression))
    except Exception:
        return "Erreur dans le calcul"


def weather_tool(city):
    return f"La météo à {city} est ensoleillée (simulation)"


def web_search_tool(query):
    return f"Résultat de recherche web pour : {query}"
