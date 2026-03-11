import re
from typing import Optional, Tuple

_EXCLUDE = {
    'What', 'When', 'Where', 'Who', 'Why', 'How', 'Which',
    'Do', 'Does', 'Did', 'Is', 'Was', 'Are', 'Were', 'Has',
    'Have', 'Had', 'Can', 'Could', 'Would', 'Should', 'Will',
    'The', 'A', 'An', 'In', 'On', 'At', 'For', 'To', 'Of',
    'Both', 'Either', 'Neither', 'Any', 'All', 'Some',
}

_Q_FRAME = re.compile(
    r'^(?:in\s+which\s+|which\s+|what\s+(?:are|is|were|was|did|does|do)?\s*|'
    r'when\s+(?:did|does|do|was|is)?\s*|where\s+(?:did|does|do|is|was)?\s*|'
    r'why\s+(?:did|does|do|is|was)?\s*|how\s+(?:did|does|do|is|was)?\s*|'
    r'who\s+(?:did|does|do|is|was)?\s*|'
    r'do(?:es)?\s+(?:both\s+\w+\s+and\s+\w+\s+)?|'
    r'did\s+|is\s+|was\s+|are\s+|were\s+)',
    re.IGNORECASE
)


def extract_name_from_query(query: str) -> Optional[str]:
    m = re.search(r"\b([A-Z][a-z]+)'s\b", query)
    if m and m.group(1) not in _EXCLUDE:
        return m.group(1)

    m = re.search(
        r'\b(?:did|does|is|was|has|have|can|could|would|should|will)\s+([A-Z][a-z]+)\b',
        query
    )
    if m and m.group(1) not in _EXCLUDE:
        return m.group(1)

    m = re.search(r'\bboth\s+([A-Z][a-z]+)\b', query)
    if m and m.group(1) not in _EXCLUDE:
        return m.group(1)

    for name in re.findall(r'\b([A-Z][a-z]{2,})\b', query):
        if name not in _EXCLUDE:
            return name

    return None


def _extract_content(query: str, name: str) -> str:
    text = query.lower()
    name_l = name.lower()

    text = _Q_FRAME.sub('', text).strip()
    text = re.sub(rf'(?:did|does|is|was|has|have)\s+{name_l}\s+', '', text, count=1)
    text = re.sub(rf"\b{name_l}'s\b", 'my', text)
    text = re.sub(rf'\b{name_l}\b', '', text)
    text = text.rstrip('?').strip()
    text = re.sub(r'\s{2,}', ' ', text)

    return text


def build_first_person_supplement(query: str, name: str) -> str:
    ql = query.lower()
    content = _extract_content(query, name)

    if ql.startswith('when'):
        return f"I {content}"
    if 'live' in ql or 'located' in ql or ql.startswith('where'):
        return f"I live in {content}"
    if ql.startswith('what') and f"{name.lower()}'s" in ql:
        return f"{content} I have {content}"
    if ql.startswith('what'):
        return f"I {content}"
    if re.match(r'^do(?:es)?\s', ql):
        return f"I {content}"
    if re.match(r'^(?:is|was|are|were)\s', ql):
        return f"I {content}"
    if re.match(r'^(?:which|in which)\s', ql):
        return f"I {content}"

    return f"I {content}"


def get_dual_queries(query: str) -> Tuple[str, Optional[str]]:
    name = extract_name_from_query(query)
    if not name:
        return query, None
    supplement = build_first_person_supplement(query, name)
    return query, f"{query} {supplement}"