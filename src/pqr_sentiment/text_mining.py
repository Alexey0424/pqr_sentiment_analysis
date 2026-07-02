"""Text mining utilities for the Spanish complaint corpus.

Provides accent tolerant normalization, a stopword list that combines
standard Spanish function words with the reporting boilerplate of this
corpus (almost every record starts with variations of "usuario manifiesta
que"), distinctive term extraction per sentiment class through TF IDF, and
simple length statistics.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ["SPANISH_STOPWORDS", "normalize_for_tokens", "top_terms_by_class", "text_length_stats"]

_CORE_STOPWORDS = """
a al algo alguna algunas alguno algunos ante antes aqui asi aun cada como con contra cual cuales
cuando de del desde donde dos durante e el ella ellas ellos en entre era eran es esa esas ese eso
esos esta estaba estaban estan este esto estos fue fueron ha haber habia han hasta hay la las le
les lo los mas me mi mientras muy nada ni no nos nuestra nuestro o os otra otras otro otros para
pero por porque que quien se segun ser si sin sobre solo son su sus tal tambien tanto tiene tienen
toda todas todo todos tras tres un una uno unas unos y ya yo
""".split()

_DOMAIN_STOPWORDS = """
usuario usuaria usuarios usuarias cliente clientes señor señora sr sra manifiesta manifiestan
solicita solicitan solicito pide piden expone exponen expresa indica indican informa informan
comunica comunican reporta reportan menciona refiere requiere dia
""".split()

SPANISH_STOPWORDS = frozenset(_CORE_STOPWORDS) | frozenset(_DOMAIN_STOPWORDS)

_KEEP_ENYE = "̃"  # combining tilde, the mark that turns n into ñ


def normalize_for_tokens(text: str) -> str:
    """Lowercase, strip accents (keeping ñ), turn punctuation into spaces."""
    lowered = text.lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    chars = []
    for index, char in enumerate(decomposed):
        if unicodedata.combining(char):
            if char == _KEEP_ENYE and index > 0 and decomposed[index - 1] == "n":
                chars.append(char)
            continue
        chars.append(char)
    stripped = unicodedata.normalize("NFC", "".join(chars))
    cleaned = re.sub(r"[^\w\sñ]", " ", stripped)
    return re.sub(r"\s+", " ", cleaned).strip()


def top_terms_by_class(
    df: pd.DataFrame,
    text_col: str,
    class_col: str,
    top_n: int = 12,
    ngram_range: tuple[int, int] = (1, 2),
) -> pd.DataFrame:
    """Highest mean TF IDF terms inside each class of ``class_col``.

    The vectorizer is fit on the whole corpus so scores are comparable, then
    terms are ranked by their mean weight within each class.
    """
    texts = df[text_col].fillna("").map(normalize_for_tokens)
    vectorizer = TfidfVectorizer(
        stop_words=sorted(SPANISH_STOPWORDS),
        ngram_range=ngram_range,
        min_df=2,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    vocabulary = vectorizer.get_feature_names_out()

    frames = []
    labels = df[class_col].reset_index(drop=True)
    for label in labels.dropna().unique():
        rows = (labels == label).to_numpy().nonzero()[0]
        means = matrix[rows].mean(axis=0).A1
        order = means.argsort()[::-1][:top_n]
        frames.append(
            pd.DataFrame(
                {"class": label, "term": vocabulary[order], "score": means[order]}
            )
        )
    result = pd.concat(frames, ignore_index=True)
    return result[result["score"] > 0].reset_index(drop=True)


def text_length_stats(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Character and word count summary for the corpus."""
    texts = df[text_col].dropna().astype(str)
    characters = texts.str.len()
    words = texts.str.split().str.len()
    return pd.DataFrame(
        {
            "min": [characters.min(), words.min()],
            "median": [characters.median(), words.median()],
            "mean": [characters.mean(), words.mean()],
            "max": [characters.max(), words.max()],
        },
        index=pd.Index(["characters", "words"], name="unit"),
    )
