import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Any, Optional


def extract_temporal_info(query: str) -> Dict[str, Any]:
    query_lower = query.lower()
    temporal_keywords = [
        'when', 'date', 'time', 'day',
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        '2020', '2021', '2022', '2023', '2024',
        'last', 'recent', 'ago', 'yesterday', 'today', 'week', 'month', 'year'
    ]
    has_temporal = any(kw in query_lower for kw in temporal_keywords)
    year_match = re.search(r'\b(20[12][0-9])\b', query)
    year = year_match.group(1) if year_match else None
    month = next((m for m in [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december'
    ] if m in query_lower), None)
    return {'has_temporal': has_temporal, 'year': year, 'month': month}


def _parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    date_match = re.search(r'(\d{1,2})\s+(\w+),?\s+(\d{4})', timestamp_str)
    if not date_match:
        return None
    date_str = f"{int(date_match.group(1))} {date_match.group(2)} {int(date_match.group(3))}"
    for fmt in ['%d %B %Y', '%d %b %Y']:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return None


def _extract_reference_date_str(timestamp_str: str) -> Optional[str]:
    date_match = re.search(r'(\d{1,2}\s+\w+),?\s+(\d{4})', timestamp_str)
    return date_match.group(0).replace(',', '') if date_match else None


def _is_already_explicit_relative(doc: str) -> bool:
    explicit_patterns = [
        r'the\s+(week|day|month|year)\s+before\s+\d{1,2}\s+\w+\s+\d{4}',
        r'the\s+(week|day|month|year)\s+after\s+\d{1,2}\s+\w+\s+\d{4}',
        r'the\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+before\s+\d{1,2}\s+\w+\s+\d{4}',
        r'the\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+after\s+\d{1,2}\s+\w+\s+\d{4}',
    ]
    return any(re.search(p, doc, re.IGNORECASE) for p in explicit_patterns)


def _resolve_absolute_terms(doc: str, ref_date: datetime) -> str:
    if re.search(r'\byesterday\b', doc, re.IGNORECASE):
        date_str = (ref_date - timedelta(days=1)).strftime('%d %B %Y')
        doc = re.sub(r'\byesterday\b', f'on {date_str}', doc, flags=re.IGNORECASE)

    if re.search(r'\btomorrow\b', doc, re.IGNORECASE):
        date_str = (ref_date + timedelta(days=1)).strftime('%d %B %Y')
        doc = re.sub(r'\btomorrow\b', f'on {date_str}', doc, flags=re.IGNORECASE)

    word_to_num = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'a': 1
    }

    for match in re.finditer(r'(\w+)\s+days?\s+ago', doc, re.IGNORECASE):
        n = word_to_num.get(match.group(1).lower())
        if n:
            date_str = (ref_date - timedelta(days=n)).strftime('%d %B %Y')
            doc = re.sub(rf'\b{match.group(1)}\s+days?\s+ago\b', f'on {date_str}',
                         doc, flags=re.IGNORECASE, count=1)

    for match in re.finditer(r'(\w+)\s+weeks?\s+ago', doc, re.IGNORECASE):
        n = word_to_num.get(match.group(1).lower())
        if n:
            date_str = (ref_date - timedelta(weeks=n)).strftime('%d %B %Y')
            doc = re.sub(rf'\b{match.group(1)}\s+weeks?\s+ago\b', f'on {date_str}',
                         doc, flags=re.IGNORECASE, count=1)

    for match in re.finditer(r'in\s+(\w+)\s+days?', doc, re.IGNORECASE):
        n = word_to_num.get(match.group(1).lower())
        if n:
            date_str = (ref_date + timedelta(days=n)).strftime('%d %B %Y')
            doc = re.sub(rf'\bin\s+{match.group(1)}\s+days?\b', f'on {date_str}',
                         doc, flags=re.IGNORECASE, count=1)

    for match in re.finditer(r'in\s+(\w+)\s+weeks?', doc, re.IGNORECASE):
        n = word_to_num.get(match.group(1).lower())
        if n:
            date_str = (ref_date + timedelta(weeks=n)).strftime('%d %B %Y')
            doc = re.sub(rf'\bin\s+{match.group(1)}\s+weeks?\b', f'on {date_str}',
                         doc, flags=re.IGNORECASE, count=1)

    return doc


def _make_relative_explicit(doc: str, ref_date: datetime, reference_date_str: str) -> str:
    if not reference_date_str:
        return doc

    weekdays = [
        ('monday', 'mon'), ('tuesday', 'tue'), ('wednesday', 'wed'),
        ('thursday', 'thu'), ('friday', 'fri'), ('saturday', 'sat'), ('sunday', 'sun')
    ]

    for full_day, abbrev in weekdays:
        pattern = rf'\blast\s+(?:{full_day}|{abbrev})\b'
        if re.search(pattern, doc, re.IGNORECASE):
            doc = re.sub(pattern, f'the {full_day} before {reference_date_str}',
                         doc, flags=re.IGNORECASE)

    if re.search(r'\blast\s+week\b', doc, re.IGNORECASE):
        doc = re.sub(r'\blast\s+week\b', f'the week before {reference_date_str}',
                     doc, flags=re.IGNORECASE)

    if re.search(r'\blast\s+weekend\b', doc, re.IGNORECASE):
        doc = re.sub(r'\blast\s+weekend\b', f'the weekend before {reference_date_str}',
                     doc, flags=re.IGNORECASE)

    if re.search(r'\blast\s+month\b', doc, re.IGNORECASE):
        month_str = (ref_date - relativedelta(months=1)).strftime('%B %Y')
        doc = re.sub(r'\blast\s+month\b', month_str, doc, flags=re.IGNORECASE)

    if re.search(r'\blast\s+year\b', doc, re.IGNORECASE):
        doc = re.sub(r'\blast\s+year\b', str(ref_date.year - 1), doc, flags=re.IGNORECASE)

    for full_day, _ in weekdays:
        pattern = rf'\bnext\s+{full_day}\b'
        if re.search(pattern, doc, re.IGNORECASE):
            doc = re.sub(pattern, f'the {full_day} after {reference_date_str}',
                         doc, flags=re.IGNORECASE)

    if re.search(r'\bnext\s+week\b', doc, re.IGNORECASE):
        doc = re.sub(r'\bnext\s+week\b', f'the week after {reference_date_str}',
                     doc, flags=re.IGNORECASE)

    if re.search(r'\bnext\s+month\b', doc, re.IGNORECASE):
        month_str = (ref_date + relativedelta(months=1)).strftime('%B %Y')
        doc = re.sub(r'\bnext\s+month\b', month_str, doc, flags=re.IGNORECASE)

    if re.search(r'\bnext\s+year\b', doc, re.IGNORECASE):
        doc = re.sub(r'\bnext\s+year\b', str(ref_date.year + 1), doc, flags=re.IGNORECASE)

    for match in re.finditer(r'in\s+(\w+)\s+months?', doc, re.IGNORECASE):
        n = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'a': 1}.get(
            match.group(1).lower())
        if n:
            month_str = (ref_date + relativedelta(months=n)).strftime('%B %Y')
            doc = re.sub(rf'\bin\s+{match.group(1)}\s+months?\b', f'in {month_str}',
                         doc, flags=re.IGNORECASE, count=1)

    if re.search(r'\bthis\s+week\b', doc, re.IGNORECASE):
        doc = re.sub(r'\bthis\s+week\b', ref_date.strftime('week of %d %B %Y'),
                     doc, flags=re.IGNORECASE)

    if re.search(r'\bthis\s+month\b', doc, re.IGNORECASE):
        doc = re.sub(r'\bthis\s+month\b', ref_date.strftime('%B %Y'),
                     doc, flags=re.IGNORECASE)

    if re.search(r'\bthis\s+year\b', doc, re.IGNORECASE):
        doc = re.sub(r'\bthis\s+year\b', str(ref_date.year), doc, flags=re.IGNORECASE)

    return doc


def resolve_dates_in_context(documents: List[str]) -> List[str]:
    resolved = []
    for doc in documents:
        if _is_already_explicit_relative(doc):
            resolved.append(doc)
            continue

        timestamp_match = re.search(r'\(([^)]+)\)', doc)
        if not timestamp_match:
            resolved.append(doc)
            continue

        ref_date = _parse_timestamp(timestamp_match.group(1))
        if not ref_date:
            resolved.append(doc)
            continue

        reference_date_str = _extract_reference_date_str(timestamp_match.group(1))
        doc = _resolve_absolute_terms(doc, ref_date)
        if reference_date_str:
            doc = _make_relative_explicit(doc, ref_date, reference_date_str)
        resolved.append(doc)

    return resolved


def normalize_answer_granularity(answer: str, query: str) -> str:
    query_lower = query.lower()

    is_past_event = any(w in query_lower for w in ['did', 'was', 'were', 'went', 'happened', 'occurred'])
    is_future_planning = any(w in query_lower for w in ['planning', 'plans', 'going to', 'will', 'gonna'])
    asks_specific_date = any(p in query_lower for p in ['what date', 'which date', 'what day', 'which day'])

    if is_future_planning and not (is_past_event or asks_specific_date):
        answer = re.sub(
            r'\bon\s+\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
            r'in \1 \2', answer, flags=re.IGNORECASE
        )
        answer = re.sub(
            r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})(?=\s*[.,]|\s*$)',
            r'\1 \2', answer, flags=re.IGNORECASE
        )

    return answer