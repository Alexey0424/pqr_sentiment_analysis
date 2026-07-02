import pandas as pd

from pqr_sentiment.text_mining import (
    SPANISH_STOPWORDS,
    normalize_for_tokens,
    text_length_stats,
    top_terms_by_class,
)


def make_corpus() -> pd.DataFrame:
    negative = [
        "la tuberia rota inunda la casa y nadie responde por los danos",
        "obras dejaron la tuberia rota frente a mi casa, exijo arreglo de los danos",
        "sigue la tuberia rota, danos en paredes y pisos por la inundacion",
        "la fuga de la tuberia rota causo danos graves en la vivienda",
    ]
    neutral = [
        "solicita informacion sobre el cronograma de pavimentacion del sector",
        "pregunta por el cronograma de obras y fechas de pavimentacion",
        "requiere informacion del cronograma para coordinar un evento",
        "consulta el cronograma de pavimentacion de la via principal",
    ]
    rows = [{"description": t, "sentiment": "NEG"} for t in negative]
    rows += [{"description": t, "sentiment": "NEU"} for t in neutral]
    return pd.DataFrame(rows)


class TestNormalize:
    def test_lowercases_and_strips_accents_but_keeps_enye(self):
        assert normalize_for_tokens("Instalación DAÑADA") == "instalacion dañada"

    def test_punctuation_becomes_spaces(self):
        assert normalize_for_tokens("obra, via; anden.") == "obra via anden"


class TestStopwords:
    def test_core_spanish_words_present(self):
        for word in ["que", "de", "la", "los", "para", "por"]:
            assert word in SPANISH_STOPWORDS

    def test_domain_boilerplate_present(self):
        for word in ["usuario", "usuaria", "manifiesta", "solicita"]:
            assert word in SPANISH_STOPWORDS


class TestTopTerms:
    def test_distinctive_terms_surface_per_class(self):
        terms = top_terms_by_class(make_corpus(), "description", "sentiment", top_n=5)
        neg_terms = terms.loc[terms["class"] == "NEG", "term"].tolist()
        neu_terms = terms.loc[terms["class"] == "NEU", "term"].tolist()
        assert any("tuberia" in t for t in neg_terms)
        assert any("cronograma" in t for t in neu_terms)

    def test_stopwords_never_appear(self):
        terms = top_terms_by_class(make_corpus(), "description", "sentiment", top_n=8)
        for term in terms["term"]:
            for token in term.split():
                assert token not in SPANISH_STOPWORDS

    def test_top_n_respected(self):
        terms = top_terms_by_class(make_corpus(), "description", "sentiment", top_n=3)
        assert terms.groupby("class").size().max() <= 3

    def test_scores_sorted_descending_within_class(self):
        terms = top_terms_by_class(make_corpus(), "description", "sentiment", top_n=5)
        for _, group in terms.groupby("class"):
            scores = group["score"].tolist()
            assert scores == sorted(scores, reverse=True)


class TestLengthStats:
    def test_stats_shape_and_values(self):
        df = pd.DataFrame({"description": ["uno dos tres", "uno"]})
        stats = text_length_stats(df, "description")
        assert stats.loc["characters", "median"] == 7.5
        assert stats.loc["words", "max"] == 3
