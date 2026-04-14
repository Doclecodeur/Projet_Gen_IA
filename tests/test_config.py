"""
Tests unitaires pour validate_config() (config.py).

Couvre :
- Clé OpenAI absente → sys.exit()
- Clés optionnelles absentes → warnings uniquement, pas d'exit
- Toutes les clés présentes → aucun warning, aucun exit

Note : load_dotenv() dans config.py est mocké pour éviter qu'il écrase
les variables d'environnement injectées par monkeypatch.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest


def _reload_config(monkeypatch, openai_key: str, tavily_key: str, owm_key: str):
    """
    Recharge config.py avec les clés spécifiées.
    load_dotenv() est neutralisé pour que monkeypatch garde le contrôle.
    """
    monkeypatch.setenv("OPENAI_API_KEY", openai_key)
    monkeypatch.setenv("TAVILY_API_KEY", tavily_key)
    monkeypatch.setenv("OPENWEATHER_API_KEY", owm_key)

    with patch("dotenv.load_dotenv"):  # empêche load_dotenv d'écraser les valeurs
        import config
        importlib.reload(config)
    return config


# ══════════════════════════════════════════════════════════════════════════════
# validate_config
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateConfig:

    def test_openai_key_manquante_fait_sys_exit(self, monkeypatch):
        """Sans OPENAI_API_KEY, validate_config doit appeler sys.exit."""
        config = _reload_config(monkeypatch, "", "tvly-test", "owm-test")
        with pytest.raises(SystemExit):
            config.validate_config()

    def test_toutes_les_cles_presentes_pas_d_erreur(self, monkeypatch, capsys):
        """Toutes les clés présentes → aucun warning, aucun exit."""
        config = _reload_config(monkeypatch, "sk-test", "tvly-test", "owm-test")
        config.validate_config()
        captured = capsys.readouterr()
        assert "absente" not in captured.out

    def test_tavily_manquante_affiche_warning(self, monkeypatch, capsys):
        """TAVILY_API_KEY absente → warning affiché, pas de sys.exit."""
        config = _reload_config(monkeypatch, "sk-test", "", "owm-test")
        config.validate_config()
        captured = capsys.readouterr()
        assert "TAVILY_API_KEY" in captured.out

    def test_openweather_manquante_affiche_warning(self, monkeypatch, capsys):
        """OPENWEATHER_API_KEY absente → warning affiché, pas de sys.exit."""
        config = _reload_config(monkeypatch, "sk-test", "tvly-test", "")
        config.validate_config()
        captured = capsys.readouterr()
        assert "OPENWEATHER_API_KEY" in captured.out

    def test_deux_cles_optionnelles_manquantes_deux_warnings(self, monkeypatch, capsys):
        """Les deux clés optionnelles absentes → deux warnings distincts."""
        config = _reload_config(monkeypatch, "sk-test", "", "")
        config.validate_config()
        captured = capsys.readouterr()
        assert "TAVILY_API_KEY" in captured.out
        assert "OPENWEATHER_API_KEY" in captured.out
