"""Privacy step: remove personal details before any modelling.

Regex for emails, URLs and phone numbers; spaCy NER (en_core_web_sm) for people and places.
"""
import re
from collections import Counter

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
URL = re.compile(r"(https?://\S+|www\.\S+|\b\w+\.(com|org|net|co\.uk|io)\b\S*)", re.I)
PHONE = re.compile(r"(?<!\w)(\+?\d[\d\s().-]{8,}\d)(?!\w)")
USER = re.compile(r"(?<!\w)/?u/[A-Za-z0-9_-]+")
NER_MAP = {"PERSON": "[PERSON]", "GPE": "[LOCATION]", "LOC": "[LOCATION]", "FAC": "[LOCATION]"}


def redact_regex(text: str, counts: Counter) -> str:
    for pat, tag, name in [(EMAIL, "[EMAIL]", "EMAIL"), (URL, "[URL]", "URL"),
                           (USER, "[USERNAME]", "USERNAME"), (PHONE, "[PHONE]", "PHONE")]:
        text, n = pat.subn(tag, text)
        counts[name] += n
    return text


def redact_ner_doc(doc, counts: Counter) -> str:
    out, last = [], 0
    for ent in doc.ents:
        if ent.label_ in NER_MAP:
            out.append(doc.text[last:ent.start_char])
            out.append(NER_MAP[ent.label_])
            counts[NER_MAP[ent.label_].strip("[]")] += 1
            last = ent.end_char
    out.append(doc.text[last:])
    return "".join(out)


def redact_all(texts, nlp=None, batch_size=64):
    counts = Counter()
    pre = [redact_regex(t, counts) for t in texts]
    if nlp is None:
        return pre, counts
    out = [redact_ner_doc(d, counts) for d in nlp.pipe(pre, batch_size=batch_size)]
    return out, counts
