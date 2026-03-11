import json
import time
import argparse
from typing import Dict, List, Any, Optional

from adaptive_rag import AdaptiveRAG
from utils.llm_utils import llm_judge


def load_locomo(filepath: str = 'locomo10.json') -> List[Dict]:
    with open(filepath) as f:
        return json.load(f)


def calculate_retrieval_metrics(retrieved_ids: List[str], evidence_ids: List[str]) -> Dict[str, float]:
    retrieved_set = set(retrieved_ids)
    evidence_set = set(evidence_ids)

    if len(evidence_set) == 0:
        recall = 1.0
    else:
        recall = len(retrieved_set & evidence_set) / len(evidence_set)

    precision = len(retrieved_set & evidence_set) / len(retrieved_set) if retrieved_set else 0.0

    mrr = 0.0
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in evidence_set:
            mrr = 1.0 / i
            break

    return {
        'recall': recall,
        'precision': precision,
        'mrr': mrr,
        'retrieved': list(retrieved_set),
        'evidence': list(evidence_set),
        'overlap': list(retrieved_set & evidence_set)
    }


def ingest_conversation(system: AdaptiveRAG, conv: Dict) -> int:
    conversation = conv['conversation']
    session_keys = sorted(
        k for k in conversation.keys()
        if k.startswith('session_') and not k.endswith('_date_time')
    )

    count = 0
    for session_key in session_keys:
        timestamp = conversation.get(f"{session_key}_date_time", "")
        for turn in conversation[session_key]:
            system.add_message(
                speaker=turn['speaker'],
                text=turn['text'],
                dia_id=turn['dia_id'],
                timestamp=timestamp,
                session=session_key
            )
            count += 1
    return count


def evaluate_conversation(
    conv_idx: int,
    conv: Dict,
    verbose: bool = False,
    max_questions: Optional[int] = None,
) -> tuple[List[Dict], Dict[str, Any]]:

    prefix = f"conv_{conv_idx}_adaptive"
    system = AdaptiveRAG(
        collection_prefix=prefix,
        working_memory_size=10,
        episodic_k=10,
        semantic_k=3,
        min_importance=0.3,
        use_reranker=True,
    )

    start = time.time()
    num_messages = ingest_conversation(system, conv)
    ingest_time = time.time() - start

    if verbose:
        print(f"Conv {conv_idx}: ingested {num_messages} messages in {ingest_time:.1f}s")

    qa_list = conv['qa'][:max_questions] if max_questions else conv['qa']

    results = []
    correct = 0
    total_recall = total_precision = total_mrr = total_latency = 0

    for i, qa in enumerate(qa_list, 1):
        question = qa['question']
        ground_truth = str(qa.get('answer', ''))
        if not ground_truth:
            continue
        evidence_ids = qa.get('evidence', [])

        result = system.answer(question, return_context=True)
        answer = result['answer']
        retrieved_ids = result['retrieved_dia_ids']
        tier_sources = result['tier_sources']
        tier_breakdown = result['tier_breakdown']

        metrics = calculate_retrieval_metrics(retrieved_ids, evidence_ids)
        is_correct, judgment = llm_judge(question, ground_truth, answer)

        if is_correct:
            correct += 1

        total_recall += metrics['recall']
        total_precision += metrics['precision']
        total_mrr += metrics['mrr']
        total_latency += result['latency']

        results.append({
            'question_num': i,
            'question': question,
            'ground_truth': ground_truth,
            'generated': answer,
            'correct': is_correct,
            'judgment': judgment,
            'latency_ms': result['latency'] * 1000,
            'recall': metrics['recall'],
            'mrr': metrics['mrr'],
            'retrieved_dia_ids': retrieved_ids,
            'evidence_ids': evidence_ids,
            'tier_sources': tier_sources,
            'tier_breakdown': tier_breakdown,
        })

        if verbose and i % 10 == 0:
            print(f"  Q{i}/{len(qa_list)} — accuracy so far: {correct/i*100:.1f}%")

    n = len(results)
    conv_metrics = {
        'conv_idx': conv_idx,
        'sample_id': conv.get('sample_id', ''),
        'num_messages': num_messages,
        'num_questions': n,
        'correct': correct,
        'accuracy': correct / n if n > 0 else 0,
        'avg_recall': total_recall / n if n > 0 else 0,
        'avg_precision': total_precision / n if n > 0 else 0,
        'avg_mrr': total_mrr / n if n > 0 else 0,
        'avg_latency_ms': total_latency / n * 1000 if n > 0 else 0,
        'ingest_time_s': ingest_time,
    }

    return results, conv_metrics


def print_summary(all_metrics: List[Dict]):
    print(f"\n{'='*70}")
    print(f"{'Conv':<6} {'Messages':<10} {'Questions':<11} {'Correct':<9} {'Accuracy':<10} {'Recall':<8} {'MRR'}")
    print(f"{'-'*70}")

    total_q = total_correct = 0
    total_recall = total_mrr = 0

    for m in all_metrics:
        print(
            f"{m['conv_idx']:<6} {m['num_messages']:<10} {m['num_questions']:<11} "
            f"{m['correct']:<9} {m['accuracy']*100:>6.1f}%   "
            f"{m['avg_recall']:.3f}   {m['avg_mrr']:.3f}"
        )
        total_q += m['num_questions']
        total_correct += m['correct']
        total_recall += m['avg_recall'] * m['num_questions']
        total_mrr += m['avg_mrr'] * m['num_questions']

    print(f"{'-'*70}")
    print(
        f"{'Total':<6} {'':<10} {total_q:<11} {total_correct:<9} "
        f"{total_correct/total_q*100:>6.1f}%   "
        f"{total_recall/total_q:.3f}   {total_mrr/total_q:.3f}"
    )
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Adaptive RAG on LoCoMo')
    parser.add_argument('--data', default='locomo10.json', help='Path to LoCoMo dataset')
    parser.add_argument('--convs', nargs='+', type=int, default=None, help='Conversation indices to evaluate (default: all)')
    parser.add_argument('--max-questions', type=int, default=None, help='Max questions per conversation')
    parser.add_argument('--output', default=None, help='Output JSON file for detailed results')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    data = load_locomo(args.data)
    conv_indices = args.convs if args.convs else list(range(len(data)))

    all_results = []
    all_metrics = []

    for idx in conv_indices:
        if idx >= len(data):
            print(f"Skipping conv {idx} — out of range")
            continue

        results, metrics = evaluate_conversation(
            conv_idx=idx,
            conv=data[idx],
            verbose=args.verbose,
            max_questions=args.max_questions,
        )
        all_results.append({'conv_idx': idx, 'results': results})
        all_metrics.append(metrics)

        print(
            f"Conv {idx}: {metrics['correct']}/{metrics['num_questions']} "
            f"({metrics['accuracy']*100:.1f}%) | "
            f"Recall={metrics['avg_recall']:.3f} | MRR={metrics['avg_mrr']:.3f}"
        )

    if len(all_metrics) > 1:
        print_summary(all_metrics)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'metrics': all_metrics, 'results': all_results}, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()