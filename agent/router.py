def route_query(query):
    query = query.lower()

    if "document" in query or "manuel" in query or "politique" in query:
        return "rag"

    elif "météo" in query or "meteo" in query or "calcul" in query:
        return "tool"

    else:
        return "chat"
