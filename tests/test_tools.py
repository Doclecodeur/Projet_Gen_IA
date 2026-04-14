"""
Tests unitaires pour les outils de l'agent.

Couvre : calculatrice (cas normaux + cas limites),
         météo (réponse API + mode démo + ville inconnue),
         todo list (lecture + écriture).
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Import différé pour éviter les effets de bord lors du chargement du module
from agent.tools import add_todo_item, calculator, get_weather, read_todo_list


# ══════════════════════════════════════════════════════════════════════════════
# Calculatrice
# ══════════════════════════════════════════════════════════════════════════════

class TestCalculator:
    """Tests de l'outil calculatrice."""

    def test_addition_simple(self):
        assert calculator.invoke("2 + 3") == "Résultat : 5"

    def test_soustraction(self):
        assert calculator.invoke("10 - 4") == "Résultat : 6"

    def test_multiplication(self):
        assert calculator.invoke("7 * 8") == "Résultat : 56"

    def test_division(self):
        result = calculator.invoke("10 / 4")
        assert result == "Résultat : 2.5"

    def test_puissance(self):
        assert calculator.invoke("2 ** 10") == "Résultat : 1024"

    def test_racine_carree(self):
        assert calculator.invoke("sqrt(144)") == "Résultat : 12.0"

    def test_expression_complexe(self):
        # sqrt(144) + 3 * (10 - 4) = 12 + 18 = 30
        assert calculator.invoke("sqrt(144) + 3 * (10 - 4)") == "Résultat : 30.0"

    def test_constante_pi(self):
        result = calculator.invoke("round(pi, 5)")
        assert result == "Résultat : 3.14159"

    def test_division_par_zero(self):
        result = calculator.invoke("1 / 0")
        assert "division par zéro" in result.lower()

    def test_expression_invalide(self):
        result = calculator.invoke("abc + xyz")
        assert "invalide" in result.lower() or "erreur" in result.lower()

    def test_injection_interdite(self):
        """Vérifie que les builtins dangereux ne sont pas accessibles."""
        result = calculator.invoke("__import__('os').system('echo pwned')")
        assert "invalide" in result.lower() or "erreur" in result.lower()

    def test_arrondi(self):
        assert calculator.invoke("round(3.14159, 2)") == "Résultat : 3.14"


# ══════════════════════════════════════════════════════════════════════════════
# Météo
# ══════════════════════════════════════════════════════════════════════════════

class TestGetWeather:
    """Tests de l'outil météo."""

    def test_mode_demo_sans_cle(self, monkeypatch):
        """Sans clé API, doit retourner des données démo sans lever d'exception."""
        monkeypatch.setenv("OPENWEATHER_API_KEY", "")
        # Recharge la config pour prendre en compte la monkeypatch
        import config
        monkeypatch.setattr(config, "OPENWEATHER_API_KEY", "")
        import agent.tools as tools_module
        monkeypatch.setattr(tools_module, "OPENWEATHER_API_KEY", "")

        result = get_weather.invoke("Paris")
        assert "[DÉMO]" in result
        assert "Paris" in result

    def test_reponse_api_reelle(self, monkeypatch):
        """Simule une réponse correcte de l'API OpenWeatherMap."""
        import agent.tools as tools_module
        monkeypatch.setattr(tools_module, "OPENWEATHER_API_KEY", "fake-key")

        fake_response = MagicMock()
        fake_response.raise_for_status = MagicMock()
        fake_response.json.return_value = {
            "main": {"temp": 15.3, "feels_like": 13.1, "humidity": 72},
            "weather": [{"description": "nuageux"}],
            "wind": {"speed": 3.5},  # m/s → 12.6 km/h
        }

        with patch("agent.tools.requests.get", return_value=fake_response):
            result = get_weather.invoke("Lyon")

        assert "Lyon" in result
        assert "15.3" in result
        assert "72%" in result
        assert "nuageux" in result.lower() or "Nuageux" in result

    def test_ville_inconnue(self, monkeypatch):
        """Une ville inexistante doit retourner un message clair, pas une exception."""
        import agent.tools as tools_module
        monkeypatch.setattr(tools_module, "OPENWEATHER_API_KEY", "fake-key")

        import requests as req_module
        fake_response = MagicMock()
        fake_response.status_code = 404
        http_error = req_module.exceptions.HTTPError(response=fake_response)
        fake_response.raise_for_status.side_effect = http_error

        with patch("agent.tools.requests.get", return_value=fake_response):
            result = get_weather.invoke("VilleQuiNExistePas")

        assert "introuvable" in result.lower() or "erreur" in result.lower()

    def test_timeout_reseau(self, monkeypatch):
        """Une erreur réseau doit être gérée proprement."""
        import agent.tools as tools_module
        import requests as req_module
        monkeypatch.setattr(tools_module, "OPENWEATHER_API_KEY", "fake-key")

        with patch("agent.tools.requests.get", side_effect=req_module.exceptions.ConnectionError("timeout")):
            result = get_weather.invoke("Paris")

        assert "impossible" in result.lower() or "erreur" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Todo list
# ══════════════════════════════════════════════════════════════════════════════

class TestTodoList:
    """Tests des outils de gestion de la liste de tâches."""

    def test_lecture_liste_existante(self):
        contenu = "- Préparer la soutenance\n- Relire le rapport\n"
        with patch("builtins.open", mock_open(read_data=contenu)):
            result = read_todo_list.invoke({})
        assert "Préparer la soutenance" in result
        assert "Relire le rapport" in result

    def test_lecture_liste_vide(self):
        with patch("builtins.open", mock_open(read_data="")):
            result = read_todo_list.invoke({})
        assert "vide" in result.lower()

    def test_lecture_fichier_absent(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = read_todo_list.invoke({})
        assert "absent" in result.lower() or "introuvable" in result.lower()

    def test_ajout_tache(self):
        m = mock_open()
        with patch("builtins.open", m):
            result = add_todo_item.invoke("Finir les tests unitaires")
        assert "Finir les tests unitaires" in result
        m().write.assert_called_once_with("- Finir les tests unitaires\n")

    def test_ajout_erreur_ecriture(self):
        with patch("builtins.open", side_effect=OSError("Permission refusée")):
            result = add_todo_item.invoke("Tâche impossible")
        assert "impossible" in result.lower() or "erreur" in result.lower()
