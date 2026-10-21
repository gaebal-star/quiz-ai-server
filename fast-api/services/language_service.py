import re


# Latin-script languages are identified from common natural-language words,
# not from isolated letters that commonly occur as geometry variables.
_LATIN_MARKERS = {
    "English": {
        "the", "of", "and", "is", "are", "was", "were", "in", "to",
        "for", "with", "from", "that", "this", "which", "on", "by",
        "as", "it", "its", "be", "has", "have", "when", "because",
    },
    "Spanish": {
        "el", "los", "las", "del", "y", "es", "son", "en", "un",
        "una", "con", "para", "por", "que", "se", "como", "está",
        "están", "sobre", "entre",
    },
    "French": {
        "le", "les", "des", "et", "est", "sont", "dans", "une", "avec",
        "pour", "par", "qui", "que", "du", "au", "aux", "sur", "ce",
        "cette", "il", "elle",
    },
    "German": {
        "der", "die", "das", "und", "ist", "sind", "von", "zu", "im",
        "mit", "für", "auf", "nicht", "eine", "einer", "den", "des",
        "ein", "welche", "wird",
    },
    "Italian": {
        "il", "lo", "gli", "della", "delle", "e", "è", "sono", "nel",
        "nella", "con", "per", "che", "una", "uno", "come", "questa",
        "questo",
    },
    "Portuguese": {
        "o", "os", "as", "do", "da", "dos", "das", "e", "é", "são",
        "em", "um", "uma", "com", "para", "por", "que", "como", "esta",
        "este",
    },
}

_SCRIPT_LANGUAGES = (
    ("Russian", re.compile(r"[\u0400-\u04FF]")),
    ("Arabic", re.compile(r"[\u0600-\u06FF]")),
    ("Hebrew", re.compile(r"[\u0590-\u05FF]")),
    ("Thai", re.compile(r"[\u0E00-\u0E7F]")),
    ("Hindi", re.compile(r"[\u0900-\u097F]")),
    ("Greek", re.compile(r"[\u0370-\u03FF]")),
)


def detect_output_language(material) -> str:
    """Choose a quiz language from extracted natural-language evidence.

    Geometry notation, variables, and isolated Latin words are not enough to
    classify a source as English. If the source language remains ambiguous,
    Korean is the default.
    """
    fields = [material.title]
    for chunk in material.chunks:
        fields.extend((chunk.topic, chunk.content))
        fields.extend(chunk.keyPoints)
        fields.extend(chunk.evidence)

    text = " ".join(value for value in fields if value)

    hangul_count = len(re.findall(r"[가-힣]", text))
    kana_count = len(re.findall(r"[\u3040-\u30ff]", text))
    han_count = len(re.findall(r"[\u4e00-\u9fff]", text))

    if kana_count >= 2:
        return "Japanese"

    if hangul_count >= 2:
        # A clearly dominant foreign-language passage can still win in mixed
        # materials; otherwise Korean text and Korean default take precedence.
        latin_tokens = re.findall(r"[a-zÀ-öø-ÿ]+", text.casefold())
        scores = {
            language: sum(token in markers for token in latin_tokens)
            for language, markers in _LATIN_MARKERS.items()
        }
        foreign_score = max(scores.values(), default=0)
        if foreign_score < 3 or hangul_count >= len("".join(latin_tokens)) * 0.35:
            return "Korean"

    if han_count >= 4 and kana_count == 0 and hangul_count == 0:
        return "Chinese"

    for language, pattern in _SCRIPT_LANGUAGES:
        if len(pattern.findall(text)) >= 4:
            return language

    tokens = re.findall(r"[a-zÀ-öø-ÿ]+", text.casefold())
    scores = {
        language: sum(token in markers for token in tokens)
        for language, markers in _LATIN_MARKERS.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    if ranked and ranked[0][1] >= 2:
        second_score = ranked[1][1] if len(ranked) > 1 else 0
        if ranked[0][1] > second_score:
            return ranked[0][0]

    return "Korean"
