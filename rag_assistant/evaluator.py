"""
Evaluation module for the Legal RAG Knowledge Assistant.
Calculates Retrieval Precision@k, Recall@k, MRR, and Answer Relevance/Groundedness.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rag_assistant.pipeline import LegalRAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class TestCaseResult:
    """Evaluation result for an individual benchmark question."""
    test_id: str
    case: str
    question: str
    target_filename: Optional[str]
    retrieved_filenames: List[str]
    ground_truth_answer: str
    generated_answer: str
    expected_keywords: List[str]
    keywords_matched: List[str]
    keyword_coverage: float
    retrieval_hit: bool  # Recall@k: was the target file retrieved?
    precision_at_k: float  # Fraction of retrieved chunks from target document
    reciprocal_rank: float  # 1 / rank of first relevant chunk
    is_negative_test: bool
    correctly_rejected: bool
    retrieval_time_sec: float
    generation_time_sec: float


@dataclass
class EvaluationSummary:
    """Aggregated evaluation metrics for a benchmark suite."""
    total_test_cases: int
    in_domain_cases: int
    out_of_domain_cases: int
    mean_precision_at_k: float
    mean_recall_at_k: float  # Hit Rate
    mean_reciprocal_rank: float  # MRR
    mean_keyword_coverage: float
    hallucination_rejection_accuracy: float
    avg_retrieval_latency_sec: float
    avg_generation_latency_sec: float
    avg_total_latency_sec: float
    details: List[TestCaseResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics summary to serializable dict."""
        return {
            "total_test_cases": self.total_test_cases,
            "in_domain_cases": self.in_domain_cases,
            "out_of_domain_cases": self.out_of_domain_cases,
            "mean_precision_at_k": round(self.mean_precision_at_k, 4),
            "mean_recall_at_k": round(self.mean_recall_at_k, 4),
            "mean_reciprocal_rank": round(self.mean_reciprocal_rank, 4),
            "mean_keyword_coverage": round(self.mean_keyword_coverage, 4),
            "hallucination_rejection_accuracy": round(self.hallucination_rejection_accuracy, 4),
            "avg_retrieval_latency_sec": round(self.avg_retrieval_latency_sec, 4),
            "avg_generation_latency_sec": round(self.avg_generation_latency_sec, 4),
            "avg_total_latency_sec": round(self.avg_total_latency_sec, 4),
        }


class RAGEvaluator:
    """Runs automated evaluation on a benchmark dataset using a live pipeline."""

    NEGATIVE_REJECTION_PHRASES = [
        "not available",
        "not found",
        "no information",
        "does not mention",
        "cannot be determined",
        "not provided",
    ]

    def __init__(self, pipeline: LegalRAGPipeline):
        self.pipeline = pipeline

    def evaluate_dataset(
        self,
        dataset_path: Union[str, Path] = "./data/evaluation_dataset.json",
        top_k: Optional[int] = None
    ) -> EvaluationSummary:
        """
        Execute evaluation across all test items in the dataset file.

        Args:
            dataset_path: JSON path containing evaluation benchmark cases.
            top_k: Top-k passages to evaluate.

        Returns:
            EvaluationSummary with detailed and aggregated metrics.
        """
        path = Path(dataset_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

        results: List[TestCaseResult] = []

        for item in items:
            test_id = item.get("id", "unknown")
            case = item.get("case", "")
            q = item.get("question", "")
            target_fn = item.get("filename")
            expected_kw = item.get("expected_keywords", [])
            ground_truth = item.get("ground_truth_answer", "")
            is_negative = target_fn is None

            # Execute query through pipeline
            pipeline_res = self.pipeline.ask(q, top_k=top_k)

            retrieved_files = [p.filename for p in pipeline_res.passages]
            gen_ans = pipeline_res.answer

            # 1. Retrieval Recall@k & Precision@k & MRR
            if is_negative:
                # For out of domain questions, retrieval isn't targeting a file
                retrieval_hit = False
                precision = 0.0
                rr = 0.0
            else:
                target_base = Path(target_fn).stem.lower()
                matches = [
                    target_base in Path(fn).stem.lower()
                    for fn in retrieved_files
                ]
                retrieval_hit = any(matches)
                precision = (
                    sum(1 for m in matches if m) / len(retrieved_files)
                    if retrieved_files else 0.0
                )
                rr = 0.0
                for rank_idx, m in enumerate(matches, start=1):
                    if m:
                        rr = 1.0 / rank_idx
                        break

            # 2. Answer Evaluation
            ans_lower = gen_ans.lower()
            if is_negative:
                correctly_rejected = any(
                    phrase in ans_lower for phrase in self.NEGATIVE_REJECTION_PHRASES
                )
                matched_kw = []
                coverage = 1.0 if correctly_rejected else 0.0
            else:
                correctly_rejected = False
                matched_kw = [
                    kw for kw in expected_kw
                    if re.search(r"\b" + re.escape(kw.lower()) + r"\b", ans_lower)
                ]
                coverage = len(matched_kw) / len(expected_kw) if expected_kw else 1.0

            results.append(
                TestCaseResult(
                    test_id=test_id,
                    case=case,
                    question=q,
                    target_filename=target_fn,
                    retrieved_filenames=retrieved_files,
                    ground_truth_answer=ground_truth,
                    generated_answer=gen_ans,
                    expected_keywords=expected_kw,
                    keywords_matched=matched_kw,
                    keyword_coverage=round(coverage, 3),
                    retrieval_hit=retrieval_hit,
                    precision_at_k=round(precision, 3),
                    reciprocal_rank=round(rr, 3),
                    is_negative_test=is_negative,
                    correctly_rejected=correctly_rejected,
                    retrieval_time_sec=pipeline_res.retrieval_time_sec,
                    generation_time_sec=pipeline_res.generation_time_sec,
                )
            )

        # Aggregate metrics
        in_domain = [r for r in results if not r.is_negative_test]
        out_domain = [r for r in results if r.is_negative_test]

        mean_precision = sum(r.precision_at_k for r in in_domain) / len(in_domain) if in_domain else 0.0
        mean_recall = sum(1.0 for r in in_domain if r.retrieval_hit) / len(in_domain) if in_domain else 0.0
        mean_mrr = sum(r.reciprocal_rank for r in in_domain) / len(in_domain) if in_domain else 0.0
        mean_kw = sum(r.keyword_coverage for r in in_domain) / len(in_domain) if in_domain else 0.0

        rejection_acc = (
            sum(1.0 for r in out_domain if r.correctly_rejected) / len(out_domain)
            if out_domain else 1.0
        )

        all_retrieval_times = [r.retrieval_time_sec for r in results]
        all_gen_times = [r.generation_time_sec for r in results]
        avg_ret_time = sum(all_retrieval_times) / len(all_retrieval_times) if all_retrieval_times else 0.0
        avg_gen_time = sum(all_gen_times) / len(all_gen_times) if all_gen_times else 0.0

        return EvaluationSummary(
            total_test_cases=len(results),
            in_domain_cases=len(in_domain),
            out_of_domain_cases=len(out_domain),
            mean_precision_at_k=mean_precision,
            mean_recall_at_k=mean_recall,
            mean_reciprocal_rank=mean_mrr,
            mean_keyword_coverage=mean_kw,
            hallucination_rejection_accuracy=rejection_acc,
            avg_retrieval_latency_sec=avg_ret_time,
            avg_generation_latency_sec=avg_gen_time,
            avg_total_latency_sec=avg_ret_time + avg_gen_time,
            details=results
        )
