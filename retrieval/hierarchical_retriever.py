from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from memory.working_memory import WorkingMemory
from memory.episodic_memory import EpisodicMemory
from memory.semantic_memory import SemanticMemory
from memory.importance_scorer import ImportanceScorer
from utils.query_rewriter import get_dual_queries


class HierarchicalRetriever:
    """Orchestrates retrieval across working, semantic, and episodic memory tiers."""

    def __init__(
        self,
        working_memory_size: int = 10,
        episodic_k: int = 5,
        semantic_k: int = 3,
        min_importance: float = 0.3,
        collection_prefix: str = "hierarchical",
        use_reranker: bool = True,
    ):
        self.working_memory = WorkingMemory(maxlen=working_memory_size)
        self.importance_scorer = ImportanceScorer()

        self.episodic_memory = EpisodicMemory(
            collection_name=f"{collection_prefix}_episodic",
            min_importance=min_importance,
            importance_scorer=self.importance_scorer,
            use_reranker=use_reranker
        )

        self.semantic_memory = SemanticMemory(
            collection_name=f"{collection_prefix}_semantic",
            min_importance_for_extraction=0.15
        )

        self.episodic_k = episodic_k
        self.semantic_k = semantic_k
        self.min_importance = min_importance
        self.stats = defaultdict(int)

    def add_message(
        self,
        speaker: str,
        text: str,
        dia_id: str,
        timestamp: Optional[str] = None,
        session: Optional[str] = None
    ) -> Dict[str, Any]:
        self.working_memory.add_turn(speaker, text, dia_id, timestamp, session)

        episodic_result = self.episodic_memory.add_message(
            speaker, text, dia_id, timestamp, session
        )
        importance_score = episodic_result['importance_score']

        facts = self.semantic_memory.extract_and_store_facts(
            speaker, text, dia_id, timestamp, session,
            importance_score=importance_score
        )

        self.stats['total_messages'] += 1
        if facts:
            self.stats['messages_with_facts'] += 1
            self.stats['total_facts'] += len(facts)

        return {
            'importance_score': importance_score,
            'dia_id': dia_id,
            'facts_extracted': len(facts)
        }

    def retrieve_hierarchical(
        self,
        query: str,
        episodic_k: Optional[int] = None,
        semantic_k: Optional[int] = None,
        include_working: bool = False,
        include_semantic: bool = True,
        include_episodic: bool = True,
        return_details: bool = False
    ) -> Dict[str, Any]:
        episodic_k = episodic_k or self.episodic_k
        semantic_k = semantic_k or self.semantic_k

        all_documents = []
        all_dia_ids = []
        all_sources = []
        seen_dia_ids: Set[str] = set()

        # 1. Working memory
        if include_working and not self.working_memory.is_empty():
            working_formatted = self.working_memory.get_formatted(include_timestamps=True)
            working_dia_ids = self.working_memory.get_dia_ids()

            for doc, dia_id in zip(working_formatted, working_dia_ids):
                if dia_id not in seen_dia_ids:
                    all_documents.append(doc)
                    all_dia_ids.append(dia_id)
                    all_sources.append('working')
                    seen_dia_ids.add(dia_id)

        # 2. Semantic memory
        if include_semantic and len(self.semantic_memory) > 0:
            semantic_results = self.semantic_memory.query_facts(
                query, k=semantic_k, return_sources=True
            )
            if semantic_results['facts']:
                for fact, source_dia_id in zip(
                    semantic_results['facts'],
                    semantic_results['source_dia_ids']
                ):
                    all_documents.append(f"[FACT] {fact}")
                    all_dia_ids.append(source_dia_id)
                    all_sources.append('semantic')

        # 3. Episodic memory — dual query with post-ranking window
        episodic_results = {'documents': [], 'dia_ids': [], 'scores': []}

        if include_episodic and len(self.episodic_memory) > 0:
            original_query, first_person_query = get_dual_queries(query)

            results_orig = self.episodic_memory.retrieve(
                original_query, k=episodic_k, return_scores=True
            )

            candidates = {}
            if results_orig['documents']:
                scores = results_orig.get('scores', [0.0] * len(results_orig['documents']))
                for doc, meta, dia_id, score in zip(
                    results_orig['documents'],
                    results_orig['metadatas'],
                    results_orig['dia_ids'],
                    scores
                ):
                    candidates[dia_id] = (doc, meta, score)

            if first_person_query:
                results_fp = self.episodic_memory.retrieve(
                    first_person_query, k=episodic_k, return_scores=True
                )
                if results_fp['documents']:
                    scores_fp = results_fp.get('scores', [0.0] * len(results_fp['documents']))
                    for doc, meta, dia_id, score in zip(
                        results_fp['documents'],
                        results_fp['metadatas'],
                        results_fp['dia_ids'],
                        scores_fp
                    ):
                        if dia_id not in candidates:
                            candidates[dia_id] = (doc, meta, score)
                        else:
                            old_doc, old_meta, old_score = candidates[dia_id]
                            candidates[dia_id] = (old_doc, old_meta, max(old_score, score))

            if candidates and self.episodic_memory.use_reranker:
                candidate_items = list(candidates.items())
                docs_for_rerank = [item[1][0] for item in candidate_items]
                rerank_scores = self.episodic_memory._rerank(query, docs_for_rerank)
                reranked = sorted(
                    zip(candidate_items, rerank_scores),
                    key=lambda x: x[1],
                    reverse=True
                )
                candidates = {
                    dia_id: (doc, meta, rerank_score)
                    for (dia_id, (doc, meta, _)), rerank_score in reranked
                }

            top_candidates = list(candidates.items())[:episodic_k]

            window_dia_ids = set()
            for dia_id, (doc, meta, score) in top_candidates[:3]:
                msg_id = meta.get('msg_id', -1)
                if msg_id >= 0:
                    for offset in [-1, 1]:
                        neighbor_id = msg_id + offset
                        if 0 <= neighbor_id < self.episodic_memory.message_count:
                            neighbor_key = f"msg_{neighbor_id}"
                            try:
                                neighbor = self.episodic_memory.collection.get(ids=[neighbor_key])
                                if neighbor['documents']:
                                    n_dia_id = neighbor['metadatas'][0]['dia_id']
                                    if n_dia_id not in candidates and n_dia_id not in window_dia_ids:
                                        window_dia_ids.add(n_dia_id)
                                        top_candidates.append((
                                            n_dia_id,
                                            (neighbor['documents'][0], neighbor['metadatas'][0], 0.0)
                                        ))
                            except:
                                pass

            for dia_id, (doc, meta, score) in top_candidates:
                if dia_id not in seen_dia_ids:
                    all_documents.append(doc)
                    all_dia_ids.append(dia_id)
                    all_sources.append('episodic')
                    seen_dia_ids.add(dia_id)
                    episodic_results['documents'].append(doc)
                    episodic_results['dia_ids'].append(dia_id)
                    episodic_results['scores'].append(score)

        result = {
            'documents': all_documents,
            'dia_ids': all_dia_ids,
            'sources': all_sources,
            'count': len(all_documents)
        }

        if return_details:
            result['details'] = {
                'working_count': sum(1 for s in all_sources if s == 'working'),
                'semantic_count': sum(1 for s in all_sources if s == 'semantic'),
                'episodic_count': sum(1 for s in all_sources if s == 'episodic'),
                'total_retrieved': len(all_documents),
                'deduplicated': len(seen_dia_ids)
            }
            if include_episodic and 'scores' in episodic_results:
                result['episodic_scores'] = episodic_results['scores']

        return result

    def clear(self):
        self.working_memory.clear()
        self.episodic_memory.clear()
        self.semantic_memory.clear()
        self.stats = defaultdict(int)

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_messages': self.stats['total_messages'],
            'messages_with_facts': self.stats['messages_with_facts'],
            'total_facts': self.stats['total_facts'],
            'working_memory': self.working_memory.get_stats(),
            'episodic_memory': self.episodic_memory.get_stats(),
            'semantic_memory': self.semantic_memory.get_stats()
        }

    def __repr__(self) -> str:
        return (
            f"HierarchicalRetriever("
            f"working={len(self.working_memory)}, "
            f"episodic={len(self.episodic_memory)}, "
            f"semantic={len(self.semantic_memory)})"
        )