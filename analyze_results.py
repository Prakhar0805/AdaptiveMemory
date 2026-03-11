"""
Analyzes evaluation results from evaluate_adaptive.py output.

Usage:
    python analyze_results.py <results_file.json>
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def classify_question_type(question: str) -> str:
    q = question.lower()
    if q.startswith('when '):   return 'temporal_when'
    if q.startswith('who '):    return 'identity_who'
    if q.startswith('where '):  return 'location_where'
    if q.startswith('what '):   return 'content_what'
    if q.startswith('how '):    return 'method_how'
    if q.startswith('why '):    return 'reason_why'
    return 'other'


def classify_failure_mode(result: Dict[str, Any]) -> str:
    if result['correct']:
        return 'success'
    recall = result['recall']
    mrr = result['mrr']
    if recall == 0.0:
        return 'retrieval_failure'
    if recall == 1.0 and mrr > 0.5:
        return 'llm_failure_good_ranking'
    if recall == 1.0:
        return 'llm_failure_poor_ranking'
    if 0.0 < recall < 1.0:
        return 'partial_retrieval'
    return 'unknown'


def _bucket(results, key_fn, buckets):
    data = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        b = key_fn(r)
        data[b]['total'] += 1
        if r['correct']:
            data[b]['correct'] += 1
    for b in data:
        t = data[b]['total']
        data[b]['accuracy'] = data[b]['correct'] / t * 100 if t else 0
    return dict(data)


def analyze_results(input_file: str) -> Dict[str, Any]:
    with open(input_file) as f:
        data = json.load(f)

    results = data['questions_and_results']
    metadata = data['metadata']
    total = len(results)
    correct_count = sum(1 for r in results if r['correct'])

    recall_buckets = _bucket(results, lambda r: (
        'recall_0.0' if r['recall'] == 0.0
        else 'recall_0.0-0.5' if r['recall'] < 0.5
        else 'recall_0.5-1.0' if r['recall'] < 1.0
        else 'recall_1.0'
    ), None)

    mrr_buckets = _bucket(results, lambda r: (
        'mrr_0.0' if r['mrr'] == 0.0
        else 'mrr_0.0-0.2' if r['mrr'] < 0.2
        else 'mrr_0.2-0.5' if r['mrr'] < 0.5
        else 'mrr_0.5+'
    ), None)

    question_types = _bucket(results, lambda r: classify_question_type(r['question']), None)
    failure_modes = _bucket(results, classify_failure_mode, None)

    retrieval_failures = [
        {
            'question_num': r['question_num'],
            'question': r['question'],
            'ground_truth': r['ground_truth'],
            'generated': r['generated'],
            'recall': r['recall'],
            'retrieved_dia_ids': r['retrieved_dia_ids'],
            'evidence_ids': r['evidence_ids'],
            'tier_breakdown': r['tier_breakdown']
        }
        for r in results if not r['correct'] and r['recall'] == 0.0
    ]

    llm_failures = [
        {
            'question_num': r['question_num'],
            'question': r['question'],
            'ground_truth': r['ground_truth'],
            'generated': r['generated'],
            'recall': r['recall'],
            'mrr': r['mrr'],
            'evidence_ids': r['evidence_ids'],
            'retrieved_dia_ids': r['retrieved_dia_ids'],
            'tier_breakdown': r['tier_breakdown'],
            'judgment': r['judgment']
        }
        for r in results if not r['correct'] and r['recall'] == 1.0
    ]

    ranking_failures = []
    for r in results:
        if 0 < r['mrr'] < 0.2:
            evidence_set = set(r['evidence_ids'])
            pos = next((i for i, d in enumerate(r['retrieved_dia_ids'], 1) if d in evidence_set), None)
            ranking_failures.append({
                'question_num': r['question_num'],
                'question': r['question'],
                'correct': r['correct'],
                'mrr': r['mrr'],
                'evidence_position': pos,
                'retrieved_dia_ids': r['retrieved_dia_ids'],
                'evidence_ids': r['evidence_ids'],
                'tier_breakdown': r['tier_breakdown']
            })

    with_sem = [r for r in results if r['tier_breakdown']['semantic'] > 0]
    without_sem = [r for r in results if r['tier_breakdown']['semantic'] == 0]

    return {
        'summary_stats': {
            'overall_accuracy': metadata['metrics']['accuracy'] * 100,
            'total_questions': total,
            'correct': correct_count,
            'incorrect': total - correct_count,
            'avg_recall': metadata['metrics']['avg_recall'],
            'avg_mrr': metadata['metrics']['avg_mrr'],
            'accuracy_by_recall': recall_buckets,
            'accuracy_by_mrr': mrr_buckets,
            'accuracy_by_question_type': question_types,
            'failure_mode_breakdown': failure_modes,
        },
        'retrieval_failures': retrieval_failures,
        'llm_failures': llm_failures,
        'ranking_failures': ranking_failures,
        'tier_analysis': {
            'with_semantic_facts': {
                'total': len(with_sem),
                'correct': sum(1 for r in with_sem if r['correct']),
                'accuracy': sum(1 for r in with_sem if r['correct']) / len(with_sem) * 100 if with_sem else 0
            },
            'without_semantic_facts': {
                'total': len(without_sem),
                'correct': sum(1 for r in without_sem if r['correct']),
                'accuracy': sum(1 for r in without_sem if r['correct']) / len(without_sem) * 100 if without_sem else 0
            }
        },
        'metadata': metadata
    }


def print_summary(report: Dict[str, Any]):
    stats = report['summary_stats']
    tier = report['tier_analysis']

    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\nOverall Performance:")
    print(f"  Total Questions: {stats['total_questions']}")
    print(f"  Correct: {stats['correct']}")
    print(f"  Incorrect: {stats['incorrect']}")
    print(f"  Accuracy: {stats['overall_accuracy']:.1f}%")
    print(f"  Avg Recall@5: {stats['avg_recall']:.3f}")
    print(f"  Avg MRR: {stats['avg_mrr']:.3f}")

    print(f"\nAccuracy by Recall Level:")
    for b, d in sorted(stats['accuracy_by_recall'].items()):
        print(f"  {b:20} : {d['correct']:3}/{d['total']:3} ({d['accuracy']:5.1f}%)")

    print(f"\nAccuracy by MRR Level:")
    for b, d in sorted(stats['accuracy_by_mrr'].items()):
        print(f"  {b:20} : {d['correct']:3}/{d['total']:3} ({d['accuracy']:5.1f}%)")

    print(f"\nAccuracy by Question Type:")
    for qt, d in sorted(stats['accuracy_by_question_type'].items(), key=lambda x: x[1]['total'], reverse=True):
        print(f"  {qt:20} : {d['correct']:3}/{d['total']:3} ({d['accuracy']:5.1f}%)")

    print(f"\nFailure Mode Breakdown:")
    for m, d in sorted(stats['failure_mode_breakdown'].items(), key=lambda x: x[1]['total'], reverse=True):
        print(f"  {m:30} : {d['total']:3} questions ({d['accuracy']:5.1f}% accuracy)")

    print(f"\nTier Analysis:")
    print(f"  With semantic facts:     {tier['with_semantic_facts']['correct']:3}/{tier['with_semantic_facts']['total']:3} ({tier['with_semantic_facts']['accuracy']:5.1f}%)")
    print(f"  Without semantic facts:  {tier['without_semantic_facts']['correct']:3}/{tier['without_semantic_facts']['total']:3} ({tier['without_semantic_facts']['accuracy']:5.1f}%)")

    print(f"\nKey Findings:")
    print(f"  • Retrieval failures: {len(report['retrieval_failures'])} questions (recall = 0.0)")
    print(f"  • LLM failures: {len(report['llm_failures'])} questions (recall = 1.0 but wrong)")
    print(f"  • Ranking failures: {len(report['ranking_failures'])} questions (MRR < 0.2)")
    print(f"  • Correct answers: {stats['correct']} questions")

    print("\n" + "=" * 80)


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <results_file.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    report = analyze_results(input_file)
    print_summary(report)

    output_file = input_file.replace('.json', '_report.json')
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {output_file}\n")


if __name__ == "__main__":
    main()