import re
import spacy
from typing import Dict, Any, Optional
from datetime import datetime
from collections import defaultdict


class ImportanceScorer:
    """Calculates importance scores for conversation messages using multiple signals."""

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 normalize_entities: bool = True, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)

        self.weights = weights or {
            'entity': 0.30, 'word_count': 0.05, 'question': 0.25,
            'temporal': 0.40, 'recency': 0.00
        }
        self.normalize_entities = normalize_entities
        self.temporal_keywords = self._build_temporal_keywords()
        self.stats = defaultdict(list)

    def _build_temporal_keywords(self) -> set:
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december',
                  'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        relative = ['yesterday', 'tomorrow', 'today', 'last week', 'next week', 'this week',
                    'last month', 'next month', 'this month', 'last year', 'next year', 'this year',
                    'ago', 'before', 'after', 'during', 'since', 'until', 'recent', 'recently',
                    'soon', 'later', 'earlier', 'the week before', 'the day before', 'the month before',
                    'the sunday before', 'the monday before', 'days ago', 'weeks ago', 'months ago',
                    'years ago', 'when', 'date', 'time', 'day']
        return set(months + days + relative)

    def calculate_importance(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        if not text or not text.strip():
            return 0.0

        metadata = metadata or {}

        entity_score = self._score_entities(text)
        word_count_score = self._score_word_count(text)
        question_score = self._score_question(text)
        temporal_score = self._score_temporal(text)
        recency_score = self._score_recency(metadata.get('timestamp'), metadata.get('reference_time'))

        entity_count = len([ent for ent in self.nlp(text).ents])

        base_score = 0.40 + (word_count_score * 0.15)
        multiplier = 1.0

        if temporal_score >= 0.9:
            multiplier *= 1.8
        elif temporal_score >= 0.5:
            multiplier *= 1.5
        elif temporal_score > 0:
            multiplier *= 1.2

        if entity_score >= 0.8:
            multiplier *= 1.6
        elif entity_score >= 0.5:
            multiplier *= 1.4
        elif entity_score > 0:
            multiplier *= 1.3

        if question_score > 0:
            multiplier *= 1.4

        if len(text.split()) <= 20 and entity_score > 0:
            multiplier *= 1.3

        if temporal_score > 0.3 and entity_count > 0:
            multiplier *= 1.4
        if temporal_score > 0.3 and question_score > 0:
            multiplier *= 1.5
        if entity_count >= 2 and temporal_score > 0.5:
            multiplier *= 1.3

        importance = min(1.0, base_score * multiplier)

        self.stats['entity_scores'].append(entity_score)
        self.stats['word_count_scores'].append(word_count_score)
        self.stats['question_scores'].append(question_score)
        self.stats['temporal_scores'].append(temporal_score)
        self.stats['recency_scores'].append(recency_score)
        self.stats['final_scores'].append(importance)

        return importance

    def _score_entities(self, text: str) -> float:
        doc = self.nlp(text)
        high_value_labels = {'PERSON', 'GPE', 'DATE', 'ORG', 'EVENT', 'FAC', 'LOC'}
        entities = [ent for ent in doc.ents if ent.label_ in high_value_labels]
        if self.normalize_entities:
            return min(len(entities) / 2.0, 1.0)
        return float(len(entities))

    def _score_word_count(self, text: str) -> float:
        wc = len(text.split())
        if wc <= 5:
            return 0.5
        elif wc <= 15:
            return 0.5 + ((wc - 5) / 10.0) * 0.25
        elif wc <= 30:
            return 0.75 + ((wc - 15) / 15.0) * 0.15
        else:
            return min(0.90 + (wc - 30) / 30.0, 1.0)

    def _score_question(self, text: str) -> float:
        stripped = text.strip()
        if stripped.endswith('?'):
            return 1.0
        lower = stripped.lower()
        for qword in ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose', 'whom']:
            if lower.startswith(qword + ' ') or lower.startswith(qword + "'"):
                return 1.0
        return 0.0

    def _score_temporal(self, text: str) -> float:
        lower = text.lower()
        for pattern in [
            r'\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',
            r'\b(?:20[12][0-9])\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        ]:
            if re.search(pattern, lower):
                return 1.0

        for expr in ['yesterday', 'tomorrow', 'today', 'last week', 'next week', 'this week',
                     'last month', 'next month', 'this month', 'last year', 'next year', 'this year',
                     'days ago', 'weeks ago', 'months ago', 'the week before', 'the day before',
                     'the sunday before']:
            if expr in lower:
                return 0.7

        for kw in ['when', 'date', 'time', 'before', 'after', 'during', 'since', 'until']:
            if kw in lower:
                return 0.4

        for month in ['january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december']:
            if month in lower:
                return 0.5

        return 0.0

    def _score_recency(self, timestamp: Optional[str], reference_time: Optional[str]) -> float:
        if not timestamp or not reference_time:
            return 0.0
        try:
            msg_time = self._parse_timestamp(timestamp)
            ref_time = self._parse_timestamp(reference_time)
            if not msg_time or not ref_time:
                return 0.0
            days_old = (ref_time - msg_time).days
            if days_old < 0:
                return 0.0
            elif days_old < 1:
                return 1.0
            elif days_old < 7:
                return 0.7
            elif days_old < 30:
                return 0.4
            return 0.0
        except:
            return 0.0

    def _parse_timestamp(self, timestamp: str) -> Optional[datetime]:
        if not timestamp:
            return None
        for fmt in ['%d %B %Y', '%d %B, %Y', '%B %Y', '%Y', '%d %b %Y', '%d %b, %Y', '%b %Y']:
            try:
                return datetime.strptime(timestamp.strip(), fmt)
            except ValueError:
                continue
        return None

    def extract_features(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)
        lower = text.lower()
        return {
            'text': text,
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'entity_count': len(doc.ents),
            'word_count': len(text.split()),
            'is_question': self._score_question(text) > 0,
            'temporal_markers': [kw for kw in self.temporal_keywords if kw in lower],
            'entity_score': self._score_entities(text),
            'word_count_score': self._score_word_count(text),
            'question_score': self._score_question(text),
            'temporal_score': self._score_temporal(text),
        }

    def get_stats(self) -> Dict[str, Any]:
        if not self.stats['final_scores']:
            return {'count': 0}
        scores = self.stats['final_scores']
        total = len(scores)
        high = sum(1 for s in scores if s > 0.7)
        medium = sum(1 for s in scores if 0.4 <= s <= 0.7)
        low = sum(1 for s in scores if s < 0.4)
        return {
            'count': total,
            'avg_entity_score': sum(self.stats['entity_scores']) / total,
            'avg_word_count_score': sum(self.stats['word_count_scores']) / total,
            'avg_question_score': sum(self.stats['question_scores']) / total,
            'avg_temporal_score': sum(self.stats['temporal_scores']) / total,
            'avg_recency_score': sum(self.stats['recency_scores']) / total,
            'avg_final_score': sum(scores) / total,
            'distribution': {
                'high (>0.7)': f"{high}/{total} ({high/total*100:.1f}%)",
                'medium (0.4-0.7)': f"{medium}/{total} ({medium/total*100:.1f}%)",
                'low (<0.4)': f"{low}/{total} ({low/total*100:.1f}%)"
            },
            'min_score': min(scores),
            'max_score': max(scores)
        }

    def reset_stats(self):
        self.stats = defaultdict(list)