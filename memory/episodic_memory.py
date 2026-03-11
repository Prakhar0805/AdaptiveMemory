import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Dict, List, Any, Optional
import re
from collections import defaultdict

from memory.importance_scorer import ImportanceScorer


class EpisodicMemory:
    """Stores conversation history with importance-weighted retrieval."""

    def __init__(self, collection_name: str = "episodic_memory",
                 embedding_model: str = 'BAAI/bge-small-en-v1.5',
                 reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                 min_importance: float = 0.3,
                 importance_scorer: Optional[ImportanceScorer] = None,
                 use_reranker: bool = True):

        self.embedding_model = SentenceTransformer(embedding_model)
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = CrossEncoder(reranker_model)

        self.importance_scorer = importance_scorer or ImportanceScorer()

        self.client = chromadb.Client()
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        self.collection = self.client.create_collection(collection_name)

        self.min_importance = min_importance
        self.message_count = 0
        self.stats = defaultdict(list)
        self.reference_time = None

    def add_message(self, speaker: str, text: str, dia_id: str,
                    timestamp: Optional[str] = None, session: Optional[str] = None,
                    **kwargs) -> Dict[str, Any]:
        if timestamp:
            self.reference_time = timestamp

        importance_score = self.importance_scorer.calculate_importance(
            text=text,
            metadata={'speaker': speaker, 'timestamp': timestamp, 'reference_time': self.reference_time}
        )

        embedding = self.embedding_model.encode(f"{speaker}: {text}")
        timestamp_str = f"({timestamp}) " if timestamp else ""
        formatted_text = f"{timestamp_str}{speaker}: {text}"

        metadata = {
            'importance_score': float(importance_score),
            'speaker': speaker,
            'dia_id': dia_id,
            'timestamp': timestamp or '',
            'session': session or '',
            'msg_id': self.message_count,
            **kwargs
        }

        self.collection.add(
            documents=[formatted_text],
            embeddings=[embedding.tolist()],
            ids=[f"msg_{self.message_count}"],
            metadatas=[metadata]
        )

        self.stats['importance_scores'].append(importance_score)
        self.stats['dia_ids'].append(dia_id)

        self.message_count += 1
        return {
            'msg_id': self.message_count - 1,
            'importance_score': importance_score,
            'dia_id': dia_id
        }

    def retrieve(self, query: str, k: int = 5, min_importance: Optional[float] = None,
                 temporal_filter: bool = True, return_scores: bool = False) -> Dict[str, List[Any]]:
        if min_importance is None:
            min_importance = self.min_importance

        query_embedding = self.embedding_model.encode(query)
        temporal_info = self._extract_temporal_info(query)

        where_clause = None
        if temporal_filter and temporal_info['has_temporal'] and temporal_info['year']:
            where_clause = {"timestamp": {"$contains": temporal_info['year']}}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(k * 4, self.message_count),
                where=where_clause
            )
        except:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(k * 4, self.message_count)
            )

        if not results['documents'] or not results['documents'][0]:
            return {'documents': [], 'metadatas': [], 'scores': [], 'dia_ids': []}

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        similarities = [1 - d for d in results['distances'][0]]

        if self.use_reranker and documents:
            rerank_scores = self._rerank(query, documents)
            if temporal_info['has_temporal']:
                rerank_scores = self._apply_temporal_boost(rerank_scores, metadatas, temporal_info)
            similarities = rerank_scores

        combined_results = []
        for doc, meta, similarity in zip(documents, metadatas, similarities):
            word_count = len(doc.split())
            if word_count <= 20:
                combined_score = similarity * 1.15
            elif word_count <= 35:
                combined_score = similarity * 1.05
            else:
                combined_score = similarity

            combined_results.append({
                'document': doc,
                'metadata': meta,
                'similarity': similarity,
                'importance': meta['importance_score'],
                'combined_score': combined_score,
                'dia_id': meta['dia_id']
            })

        filtered = [r for r in combined_results if r['importance'] >= min_importance]
        filtered.sort(key=lambda x: x['combined_score'], reverse=True)
        top_k = filtered[:k]

        result_dict = {
            'documents': [r['document'] for r in top_k],
            'metadatas': [r['metadata'] for r in top_k],
            'dia_ids': [r['dia_id'] for r in top_k]
        }

        if return_scores:
            result_dict['scores'] = [r['combined_score'] for r in top_k]
            result_dict['importance_scores'] = [r['importance'] for r in top_k]
            result_dict['similarity_scores'] = [r['similarity'] for r in top_k]

        return result_dict

    def retrieve_with_context_window(self, query: str, k: int = 5,
                                     window_size: int = 2,
                                     min_importance: Optional[float] = None,
                                     temporal_filter: bool = True,
                                     return_scores: bool = False) -> Dict[str, List[Any]]:
        core_results = self.retrieve(
            query, k=k, min_importance=min_importance,
            temporal_filter=temporal_filter, return_scores=True
        )

        if not core_results['documents']:
            return core_results

        core_msg_ids = [meta['msg_id'] for meta in core_results['metadatas']]
        all_msg_ids = set(core_msg_ids)
        for msg_id in core_msg_ids:
            for offset in range(-window_size, window_size + 1):
                windowed_id = msg_id + offset
                if 0 <= windowed_id < self.message_count:
                    all_msg_ids.add(windowed_id)

        try:
            windowed_results = self.collection.get(
                ids=[f"msg_{mid}" for mid in sorted(all_msg_ids)]
            )
        except:
            return core_results

        combined = sorted(
            zip(windowed_results['documents'], windowed_results['metadatas']),
            key=lambda x: x[1]['msg_id']
        )

        result_dict = {
            'documents': [doc for doc, _ in combined],
            'metadatas': [meta for _, meta in combined],
            'dia_ids': [meta['dia_id'] for _, meta in combined]
        }

        if return_scores:
            scores = []
            for meta in result_dict['metadatas']:
                if meta['msg_id'] in core_msg_ids:
                    idx = core_msg_ids.index(meta['msg_id'])
                    scores.append(core_results['scores'][idx])
                else:
                    scores.append(0.0)
            result_dict['scores'] = scores
            result_dict['importance_scores'] = [m['importance_score'] for m in result_dict['metadatas']]
            result_dict['similarity_scores'] = [0.0] * len(result_dict['documents'])

        return result_dict

    def _rerank(self, query: str, documents: List[str]) -> List[float]:
        pairs = [[query, doc] for doc in documents]
        return self.reranker.predict(pairs).tolist()

    def _apply_temporal_boost(self, scores: List[float], metadatas: List[Dict],
                              temporal_info: Dict[str, Any]) -> List[float]:
        boosted = []
        for score, meta in zip(scores, metadatas):
            timestamp = meta.get('timestamp', '')
            if temporal_info['year'] and temporal_info['year'] in timestamp:
                score *= 1.5
            if temporal_info['month'] and temporal_info['month'].lower() in timestamp.lower():
                score *= 1.3
            boosted.append(score)
        return boosted

    def _extract_temporal_info(self, query: str) -> Dict[str, Any]:
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

    def get_importance_distribution(self) -> Dict[str, Any]:
        if not self.stats['importance_scores']:
            return {'count': 0}
        scores = self.stats['importance_scores']
        total = len(scores)
        high = sum(1 for s in scores if s > 0.7)
        medium = sum(1 for s in scores if 0.4 <= s <= 0.7)
        low = sum(1 for s in scores if s < 0.4)
        return {
            'count': total,
            'avg_importance': sum(scores) / total,
            'distribution': {
                'high (>0.7)': f"{high}/{total} ({high/total*100:.1f}%)",
                'medium (0.4-0.7)': f"{medium}/{total} ({medium/total*100:.1f}%)",
                'low (<0.4)': f"{low}/{total} ({low/total*100:.1f}%)"
            },
            'min_score': min(scores),
            'max_score': max(scores)
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_messages': self.message_count,
            'importance_distribution': self.get_importance_distribution(),
            'collection_size': self.collection.count(),
            'reference_time': self.reference_time
        }

    def clear(self):
        try:
            self.client.delete_collection(self.collection.name)
        except:
            pass
        self.collection = self.client.create_collection(self.collection.name)
        self.message_count = 0
        self.stats = defaultdict(list)
        self.reference_time = None

    def __len__(self) -> int:
        return self.message_count

    def __repr__(self) -> str:
        return f"EpisodicMemory(messages={self.message_count}, min_importance={self.min_importance})"