import re
import spacy
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Optional
from collections import defaultdict


class SemanticMemory:
    """Stores extracted facts for direct knowledge retrieval."""

    def __init__(self, collection_name: str = "semantic_facts",
                 embedding_model: str = 'BAAI/bge-small-en-v1.5',
                 spacy_model: str = "en_core_web_sm",
                 min_importance_for_extraction: float = 0.15):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)

        self.embedding_model = SentenceTransformer(embedding_model)

        self.client = chromadb.Client()
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        self.collection = self.client.create_collection(collection_name)

        self.min_importance = min_importance_for_extraction
        self.fact_count = 0
        self.stats = defaultdict(list)
        self.fact_patterns = self._build_fact_patterns()

    def _build_fact_patterns(self) -> List[Dict[str, Any]]:
        return [
            {'pattern': r'(\w+)\s+(?:went|visited|attended)\s+([^,\.]+?)\s+on\s+(\d{1,2}\s+\w+\s+\d{4})',
             'fact_type': 'temporal_event'},
            {'pattern': r'(\w+)\s+(?:is planning|planning)\s+(?:to\s+)?(?:visit|go to|attend)\s+([^,\.]+?)\s+in\s+(\w+\s+\d{4})',
             'fact_type': 'temporal_plan'},
            {'pattern': r'(\w+)\s+(?:went|visited)\s+([^,\.]+?)\s+(?:yesterday|last week|last month)',
             'fact_type': 'past_event'},
            {'pattern': r'(\w+)\s+(?:went to|visited|attended)\s+([^,\.]+)',
             'fact_type': 'activity'},
            {'pattern': r'(\w+)\s+(?:ran|completed|participated in)\s+(?:a\s+)?([^,\.]+)',
             'fact_type': 'activity'},
            {'pattern': r'(\w+)\s+(?:with|alongside)\s+(?:my|her|his|their)\s+(\w+)',
             'fact_type': 'relationship'},
            {'pattern': r'(\w+)\'s\s+(\w+)', 'fact_type': 'possession'},
            {'pattern': r'(\w+)\s+is\s+(?:a\s+)?([^,\.]+?)(?:\s+woman|\s+man)?',
             'fact_type': 'identity'},
            {'pattern': r'(?:planning|plans)\s+to\s+([^,\.]+)',
             'fact_type': 'future_plan'},
        ]

    def extract_and_store_facts(self, speaker: str, text: str, dia_id: str,
                                timestamp: Optional[str] = None, session: Optional[str] = None,
                                importance_score: Optional[float] = None) -> List[str]:
        if importance_score is not None and importance_score < self.min_importance:
            return []

        facts = list(set(self._extract_facts(speaker, text, timestamp)))
        stored = []
        for fact_text in facts:
            self._store_fact(fact_text, dia_id, speaker, timestamp, session)
            stored.append(fact_text)
            self.stats['facts'].append(fact_text)
            self.stats['source_dia_ids'].append(dia_id)

        return stored

    def _extract_facts(self, speaker: str, text: str, timestamp: Optional[str] = None) -> List[str]:
        facts = []
        facts.extend(self._extract_entity_facts(speaker, text))
        facts.extend(self._extract_pattern_facts(speaker, text))
        facts.extend(self._extract_temporal_facts(speaker, text, timestamp))
        facts.extend(self._extract_relationship_facts(speaker, text))
        return facts

    def _extract_entity_facts(self, speaker: str, text: str) -> List[str]:
        doc = self.nlp(text)
        facts = []
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                facts.append(f"{ent.text} mentioned by {speaker}")
            elif ent.label_ == 'GPE':
                facts.append(f"{ent.text} location mentioned")
            elif ent.label_ == 'DATE':
                facts.append(f"Date: {ent.text}")
            elif ent.label_ == 'ORG':
                facts.append(f"{ent.text} organization mentioned")
            elif ent.label_ == 'EVENT':
                facts.append(f"Event: {ent.text}")
        return facts

    def _extract_pattern_facts(self, speaker: str, text: str) -> List[str]:
        facts = []
        for p in self.fact_patterns:
            for match in re.finditer(p['pattern'], text, re.IGNORECASE):
                ft = p['fact_type']
                if ft == 'temporal_event':
                    facts.append(f"{match.group(1)} visited {match.group(2)} on {match.group(3)}")
                elif ft == 'temporal_plan':
                    facts.append(f"{match.group(1)} planning to visit {match.group(2)} in {match.group(3)}")
                elif ft == 'activity':
                    facts.append(f"{match.group(1)} visited {match.group(2)}")
                elif ft == 'relationship':
                    facts.append(f"{speaker} with {match.group(2)}")
                elif ft == 'identity':
                    facts.append(f"{match.group(1)} is {match.group(2)}")
                elif ft == 'future_plan':
                    facts.append(f"{speaker} planning to {match.group(1)}")
        return facts

    def _extract_temporal_facts(self, speaker: str, text: str, timestamp: Optional[str] = None) -> List[str]:
        facts = []
        date_patterns = [
            (r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b', 'full_date'),
            (r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b', 'month_year'),
            (r'\b(20[12][0-9])\b', 'year'),
        ]
        for pattern, date_type in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                date = match.group(1)
                if date_type == 'full_date':
                    facts.append(f"Event on {date}")
                    facts.append(f"{speaker}: Event on {date}")
                elif date_type == 'month_year':
                    facts.append(f"Event in {date}")
                    facts.append(f"{speaker}: Event in {date}")
                elif date_type == 'year':
                    facts.append(f"Event in year {date}")
                    facts.append(f"{speaker}: Event in year {date}")
        return facts

    def _extract_relationship_facts(self, speaker: str, text: str) -> List[str]:
        facts = []
        lower = text.lower()
        for pattern in [
            r'(?:with\s+)?(?:my|her|his|their)\s+(sister|brother|mother|father|parent|sibling)',
            r'(sister|brother|mother|father|parent|sibling)',
            r'(girlfriend|boyfriend|partner|friend)',
            r'(?:with\s+)?(?:my|her|his|their)\s+(friend)',
            r'\b(single|married|dating|divorced)\b',
        ]:
            for match in re.finditer(pattern, lower):
                facts.append(f"{speaker} has {match.group(1)}" if 'single' not in match.group(1)
                             else f"{speaker} is {match.group(1)}")
        return facts

    def _store_fact(self, fact_text: str, source_dia_id: str, speaker: str,
                    timestamp: Optional[str] = None, session: Optional[str] = None):
        embedding = self.embedding_model.encode(fact_text)
        self.collection.add(
            documents=[fact_text],
            embeddings=[embedding.tolist()],
            ids=[f"fact_{self.fact_count}"],
            metadatas=[{
                'fact_text': fact_text,
                'source_dia_id': source_dia_id,
                'speaker': speaker,
                'timestamp': timestamp or '',
                'session': session or '',
                'fact_id': self.fact_count
            }]
        )
        self.fact_count += 1

    def query_facts(self, query: str, k: int = 3, return_sources: bool = False) -> Dict[str, List[Any]]:
        if self.fact_count == 0:
            return {'facts': [], 'metadatas': [], 'source_dia_ids': []}

        query_embedding = self.embedding_model.encode(query)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, self.fact_count)
        )

        if not results['documents'] or not results['documents'][0]:
            return {'facts': [], 'metadatas': [], 'source_dia_ids': []}

        result_dict = {
            'facts': results['documents'][0],
            'metadatas': results['metadatas'][0]
        }
        if return_sources:
            result_dict['source_dia_ids'] = [m['source_dia_id'] for m in results['metadatas'][0]]

        return result_dict

    def get_all_facts(self) -> List[Dict[str, Any]]:
        if self.fact_count == 0:
            return []
        results = self.collection.get()
        return [
            {'fact_text': doc, 'source_dia_id': meta['source_dia_id'],
             'speaker': meta['speaker'], 'timestamp': meta.get('timestamp', '')}
            for doc, meta in zip(results['documents'], results['metadatas'])
        ]

    def get_stats(self) -> Dict[str, Any]:
        unique_sources = len(set(self.stats['source_dia_ids'])) if self.stats['source_dia_ids'] else 0
        return {
            'total_facts': self.fact_count,
            'facts_per_message': len(self.stats['facts']) / max(unique_sources, 1),
            'unique_sources': unique_sources,
            'sample_facts': self.stats['facts'][:5] if self.stats['facts'] else []
        }

    def clear(self):
        try:
            self.client.delete_collection(self.collection.name)
        except:
            pass
        self.collection = self.client.create_collection(self.collection.name)
        self.fact_count = 0
        self.stats = defaultdict(list)

    def __len__(self) -> int:
        return self.fact_count

    def __repr__(self) -> str:
        return f"SemanticMemory(facts={self.fact_count}, min_importance={self.min_importance})"