from typing import Dict, List, Any, Optional
import time
import re

from retrieval.hierarchical_retriever import HierarchicalRetriever
from utils.date_utils import resolve_dates_in_context, normalize_answer_granularity
from utils.llm_utils import call_llm


class AdaptiveRAG:
    """Hierarchical memory RAG system for long-horizon conversational QA."""

    def __init__(
        self,
        collection_prefix: str = "adaptive",
        working_memory_size: int = 10,
        episodic_k: int = 5,
        semantic_k: int = 3,
        min_importance: float = 0.3,
        use_reranker: bool = True,
        model: str = 'qwen2.5:7b'
    ):
        self.retriever = HierarchicalRetriever(
            working_memory_size=working_memory_size,
            episodic_k=episodic_k,
            semantic_k=semantic_k,
            min_importance=min_importance,
            collection_prefix=collection_prefix,
            use_reranker=use_reranker,
        )
        self.model = model
        self.episodic_k = episodic_k
        self.semantic_k = semantic_k
        self.message_count = 0

    def add_message(
        self,
        speaker: str,
        text: str,
        dia_id: str,
        timestamp: Optional[str] = None,
        session: Optional[str] = None
    ) -> Dict[str, Any]:
        result = self.retriever.add_message(speaker, text, dia_id, timestamp, session)
        self.message_count += 1
        return result

    def answer(
        self,
        query: str,
        k: Optional[int] = None,
        return_context: bool = False
    ) -> Dict[str, Any]:
        start_time = time.time()

        retrieval_results = self.retriever.retrieve_hierarchical(
            query,
            episodic_k=k or self.episodic_k,
            semantic_k=self.semantic_k,
            include_working=False,
            include_semantic=True,
            include_episodic=True,
            return_details=True
        )

        raw_documents = retrieval_results['documents']
        dia_ids = retrieval_results['dia_ids']
        tier_sources = retrieval_results['sources']

        resolved_documents = resolve_dates_in_context(raw_documents)
        context = self._format_tiered_context(resolved_documents, tier_sources, query)
        prompt = self._build_prompt(context, query)

        llm_result = call_llm(prompt, model=self.model)
        answer = llm_result['answer']
        answer = normalize_answer_granularity(answer, query)
        answer = self._normalize_date_format(answer)

        result = {
            'answer': answer,
            'latency': time.time() - start_time,
            'retrieved_dia_ids': dia_ids,
            'tier_sources': tier_sources,
            'tier_breakdown': {
                'working': sum(1 for s in tier_sources if s == 'working'),
                'semantic': sum(1 for s in tier_sources if s == 'semantic'),
                'episodic': sum(1 for s in tier_sources if s == 'episodic')
            }
        }

        if return_context:
            result['retrieved_docs'] = resolved_documents
            result['raw_context'] = context

        return result

    def _format_tiered_context(
        self,
        documents: List[str],
        tier_sources: List[str],
        query: str
    ) -> str:
        temporal_keywords = ['when', 'what date', 'what time', 'which day', 'which month', 'which year', 'what year']
        is_temporal = any(kw in query.lower() for kw in temporal_keywords)

        if not is_temporal:
            cleaned = []
            for doc, tier in zip(documents, tier_sources):
                if tier == 'episodic':
                    cleaned.append(re.sub(r'^\([^)]+\)\s+', '', doc))
                else:
                    cleaned.append(doc)
            documents = cleaned

        working_docs = [doc for doc, src in zip(documents, tier_sources) if src == 'working']
        semantic_docs = [doc for doc, src in zip(documents, tier_sources) if src == 'semantic']
        episodic_docs = [doc for doc, src in zip(documents, tier_sources) if src == 'episodic']

        parts = []

        if working_docs:
            parts.append("=== Recent Context (Working Memory) ===")
            parts.extend(working_docs)
            parts.append("")

        if semantic_docs:
            parts.append("=== Relevant Facts (Semantic Memory) ===")
            for doc in semantic_docs:
                parts.append(doc.replace("[FACT] ", "• "))
            parts.append("")

        if episodic_docs:
            parts.append("=== Additional Context (Episodic Memory) ===")
            parts.extend(episodic_docs)

        return "\n".join(parts)

    def _normalize_date_format(self, answer: str) -> str:
        answer = re.sub(r'\b0(\d)\s+', r'\1 ', answer)
        answer = re.sub(r'(\d{1,2}\s+\w+),\s+(\d{4})', r'\1 \2', answer)
        return answer

    def _build_prompt(self, context: str, query: str) -> str:
        query_lower = query.lower()

        is_temporal = any(kw in query_lower for kw in [
            'when', 'what date', 'what time', 'which day', 'which month', 'which year'
        ])
        is_factual_entity = any(kw in query_lower for kw in [
            'which game', 'what game', 'what book', 'what song',
            'what movie', 'what show', 'what sport'
        ])
        is_yesno = query_lower.startswith((
            'does', 'did', 'is', 'was', 'are', 'were',
            'has', 'have', 'had', 'do', 'can', 'could'
        ))

        if is_temporal:
            instructions = """INSTRUCTIONS:
- Read ALL context sections carefully before answering.
- Find the specific date or time period mentioned in relation to this question.
- Answer ONLY with the date/time requested — no extra explanation.
- Use the EXACT format from the conversation (e.g. "February 2022", "first week of April 2022").
- Do NOT convert relative dates to absolute dates.
- Do NOT say "I don't know" if a date appears anywhere in the context — extract it."""

        elif is_factual_entity:
            instructions = """INSTRUCTIONS:
- Read ALL context sections carefully before answering.
- Look for the SPECIFIC name, word, or term being asked about.
- Answer with the exact name or term from the conversation — do not paraphrase or describe it.
- Keep your answer short — one word or phrase is usually enough.
- Only say "I don't know" if the specific term genuinely does not appear anywhere in the context."""

        elif is_yesno:
            instructions = """INSTRUCTIONS:
- Read ALL context sections carefully before answering.
- Answer with Yes, No, or a qualified answer (e.g. "Likely yes") based on evidence.
- If the answer is implied but not stated directly, make a reasonable inference and state it briefly.
- Keep your answer to one sentence.
- Only say "I don't know" if there is truly no relevant information anywhere in the context."""

        else:
            instructions = """INSTRUCTIONS:
- Read ALL context sections carefully before answering.
- Answer the question directly and concisely based on the conversation history.
- Use specific details from the context rather than vague descriptions.
- Keep your answer to 1-2 sentences unless more detail is genuinely needed.
- Only say "I don't know" if there is truly no relevant information anywhere in the context."""

        return f"""You are answering questions about a conversation history. \
The context below contains relevant excerpts. Read every section carefully.

{context}

Question: {query}

{instructions}

Answer:"""

    def get_stats(self) -> Dict[str, Any]:
        stats = self.retriever.get_stats()
        stats['message_count'] = self.message_count
        return stats

    def clear(self):
        self.retriever.clear()
        self.message_count = 0

    def __repr__(self) -> str:
        return f"AdaptiveRAG(messages={self.message_count})"