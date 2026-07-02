import json

import pandas as pd
import pytest

from pqr_sentiment.anonymize import TermRule, load_term_rules, scrub_series, scrub_text


class TestBuiltinPatterns:
    def test_mobile_phone_is_masked(self):
        clean, hits = scrub_text("Contactar al 3196627920 en la tarde", [])
        assert "3196627920" not in clean
        assert "[TELEFONO]" in clean
        assert hits["mobile_phone"] == 1

    def test_landline_is_masked(self):
        clean, _ = scrub_text("El fijo es 4674389, llamar en la noche", [])
        assert "4674389" not in clean
        assert "[TELEFONO]" in clean

    def test_document_number_is_masked(self):
        clean, hits = scrub_text("Cedula 42677252 del solicitante", [])
        assert "42677252" not in clean
        assert "[DOCUMENTO]" in clean
        assert hits["document"] == 1

    def test_street_address_is_masked(self):
        clean, _ = scrub_text("Vive en la CL 52 # 47-46 hace anos", [])
        assert "52" not in clean
        assert "[DIRECCION]" in clean

    def test_spelled_address_is_masked(self):
        clean, _ = scrub_text("sobre la calle 53 cerca del parque", [])
        assert "calle 53" not in clean
        assert "[DIRECCION]" in clean

    def test_vehicle_plate_is_masked(self):
        clean, _ = scrub_text("El camion de placa TMX 381 estaba alli", [])
        assert "TMX" not in clean
        assert "[PLACA]" in clean

    def test_lowercase_and_dashed_plates_are_masked(self):
        clean, _ = scrub_text("el vehiculo wcu315 y el camion de placas WCU – 315", [])
        assert "wcu" not in clean.lower()
        assert clean.count("[PLACA]") == 2

    def test_address_does_not_swallow_following_word(self):
        clean, _ = scrub_text("zona verde ubicada en diagonal 45 con avenida 38, cerca", [])
        assert "[DIRECCION] con [DIRECCION]" in clean

    def test_lowercase_words_before_numbers_are_not_plates(self):
        clean, _ = scrub_text("faltan aun 300 metros de red por instalar", [])
        assert "aun 300 metros" in clean

    def test_bare_house_number_is_masked(self):
        clean, _ = scrub_text("la tapa de registro de la vecina esta inestable # 53-23 en la via", [])
        assert "53-23" not in clean
        assert "[DIRECCION]" in clean

    def test_year_is_not_masked_as_document(self):
        clean, _ = scrub_text("Sucedio en el 2018 durante las obras", [])
        assert "2018" in clean

    def test_small_quantities_are_not_masked(self):
        clean, _ = scrub_text("Se afectaron 9 viviendas y 45 metros de red", [])
        assert "9 viviendas" in clean
        assert "45 metros" in clean


class TestTermRules:
    def test_literal_term_masked_case_insensitive(self):
        rules = [TermRule(pattern="Acme", replacement="[EMPRESA]")]
        clean, hits = scrub_text("ACME y acme S.A. son la misma Acme", rules)
        assert "acme" not in clean.lower()
        assert clean.count("[EMPRESA]") == 3
        assert hits["term:Acme"] == 3

    def test_literal_term_respects_word_boundaries(self):
        rules = [TermRule(pattern="Rio", replacement="[EMPRESA]")]
        clean, _ = scrub_text("El barrio Riofrio queda lejos de Rio", rules)
        assert "Riofrio" in clean
        assert clean.endswith("[EMPRESA]")

    def test_regex_term_rule(self):
        rules = [TermRule(pattern=r"Consorcio\s+\w+", replacement="[CONTRATISTA]", regex=True)]
        clean, _ = scrub_text("El Consorcio Andino ejecuta la obra", rules)
        assert "Consorcio" not in clean
        assert "[CONTRATISTA]" in clean

    def test_accented_variants_of_literal_are_masked(self):
        rules = [TermRule(pattern="Corbatán", replacement="[ORGANIZACION]")]
        clean, _ = scrub_text("La corporacion Corbatan y Corbatán", rules)
        assert "orbat" not in clean

    def test_scrub_is_idempotent_on_placeholders(self):
        rules = [TermRule(pattern="Acme", replacement="[EMPRESA]")]
        once, _ = scrub_text("Acme dano la CL 52 # 47-46, tel 3196627920", rules)
        twice, hits = scrub_text(once, rules)
        assert once == twice
        assert sum(hits.values()) == 0


class TestSeriesAndConfig:
    def test_scrub_series_aggregates_hits(self):
        s = pd.Series(["Llamar al 3196627920", "Sin datos personales", None])
        clean, hits = scrub_series(s, [])
        assert clean[0] == "Llamar al [TELEFONO]"
        assert clean[1] == "Sin datos personales"
        assert pd.isna(clean[2])
        assert hits["mobile_phone"] == 1

    def test_load_term_rules_from_json(self, tmp_path):
        payload = {
            "terms": [
                {"pattern": "Acme", "replacement": "[EMPRESA]"},
                {"pattern": "Consorcio\\s+\\w+", "replacement": "[CONTRATISTA]", "regex": True},
            ]
        }
        path = tmp_path / "terms.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        rules = load_term_rules(path)
        assert len(rules) == 2
        assert rules[0] == TermRule(pattern="Acme", replacement="[EMPRESA]", regex=False)
        assert rules[1].regex is True

    def test_load_term_rules_rejects_bad_payload(self, tmp_path):
        path = tmp_path / "terms.json"
        path.write_text(json.dumps({"nope": []}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_term_rules(path)
