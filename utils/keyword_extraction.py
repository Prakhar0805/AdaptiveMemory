import re
import spacy
from typing import List, Set

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

_STOPWORDS = {'have', 'make', 'take', 'give', 'go', 'come', 'know', 'think', 'want'}


def extract_keywords(query: str, documents: List[str], max_keywords: int = 5) -> Set[str]:
    keywords = set()

    query_doc = nlp(query.lower())
    for token in query_doc:
        if token.ent_type_ in ['PERSON', 'ORG', 'EVENT', 'WORK_OF_ART']:
            keywords.add(token.text.lower())
        elif token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 3:
            keywords.add(token.text.lower())

    combined = re.sub(r'\([^)]+\)\s*', '', ' '.join(documents[:3]))
    combined = re.sub(r'\w+:\s*', '', combined)

    word_freq: dict = {}
    for token in nlp(combined.lower()):
        if token.ent_type_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'DATE']:
            word_freq[token.text.lower()] = word_freq.get(token.text.lower(), 0) + 2
        elif token.pos_ in ['NOUN', 'PROPN', 'VERB'] and len(token.text) > 3:
            lemma = token.lemma_.lower()
            if lemma not in _STOPWORDS:
                word_freq[lemma] = word_freq.get(lemma, 0) + 1

    top = [kw for kw, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]]
    keywords.update(top)
    keywords -= set(query.lower().split())

    return keywords


def expand_query(original_query: str, keywords: Set[str]) -> str:
    extras = list(keywords)[:4]
    return original_query + (' ' + ' '.join(extras) if extras else '')