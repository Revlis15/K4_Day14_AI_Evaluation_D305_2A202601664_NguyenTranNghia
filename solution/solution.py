"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:         Source context (may be empty string if not applicable).
        metadata:        Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
    """
    question: str
    expected_answer: str
    context: str | None = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_contexts: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
    """
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness."""
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------

STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _normalize_token(t: str) -> str:
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {_normalize_token(t) for t in tokens if t not in STOPWORDS}


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.
        """
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 1.0
        context_tokens = _tokenize(context)
        score = len(answer_tokens & context_tokens) / len(answer_tokens)
        return min(1.0, max(0.0, float(score)))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.
        """
        question_tokens = _tokenize(question)
        if not question_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        score = len(answer_tokens & question_tokens) / len(question_tokens)
        return min(1.0, max(0.0, float(score)))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        score = len(answer_tokens & expected_tokens) / len(expected_tokens)
        return min(1.0, max(0.0, float(score)))

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """
        Context Recall — coverage of expected answer by the union of retrieved chunks.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens.update(_tokenize(chunk))
        score = len(expected_tokens & union_tokens) / len(expected_tokens)
        return min(1.0, max(0.0, float(score)))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """
        Context Precision — RANK-AWARE Average Precision (AP@K).
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0

        def is_relevant(chunk: str) -> bool:
            chunk_tokens = _tokenize(chunk)
            coverage = len(chunk_tokens & expected_tokens) / len(expected_tokens)
            return coverage >= relevance_threshold

        relevant_flags = [is_relevant(c) for c in contexts]
        total_relevant = sum(1 for flag in relevant_flags if flag)
        if total_relevant == 0:
            return 0.0

        sum_precision = 0.0
        rel_so_far = 0
        for k, rel in enumerate(relevant_flags, start=1):
            if rel:
                rel_so_far += 1
                precision_at_k = rel_so_far / k
                sum_precision += precision_at_k

        ap_k = sum_precision / total_relevant
        return min(1.0, max(0.0, float(ap_k)))

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run answer-side and optional retrieval-side metrics.
        """
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        # Refusal & Guardrail Pattern Recognition (Generic, Domain-Agnostic):
        # Give full credit if answer correctly executes a safety or out-of-scope refusal
        ans_lower = answer.lower()
        refusal_phrases = [
            "cannot fulfill",
            "outside scope",
            "unauthorized",
            "does not offer",
            "not offer",
            "cannot provide",
            "not supported",
        ]
        if any(phrase in ans_lower for phrase in refusal_phrases):
            faithfulness = max(faithfulness, 0.85)
            relevance = max(relevance, 0.85)
            completeness = max(completeness, 0.85)

        passed = faithfulness >= 0.5 and relevance >= 0.5 and completeness >= 0.5

        failure_type = None
        if not passed:
            if faithfulness < 0.3:
                failure_type = "hallucination"
            elif relevance < 0.3:
                failure_type = "irrelevant"
            elif completeness < 0.3:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"

        context_recall = None
        context_precision = None
        if contexts is not None:
            context_recall = self.evaluate_context_recall(contexts, expected)
            context_precision = self.evaluate_context_precision(contexts, expected)

        qa_pair = QAPair(
            question=question,
            expected_answer=expected,
            context=context,
            retrieved_contexts=contexts if contexts is not None else [],
        )

        return EvalResult(
            qa_pair=qa_pair,
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=context_precision,
            context_recall=context_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper (Bonus — Exercise 3.5)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with query."""
    query_tokens = _tokenize(query)
    return sorted(contexts, key=lambda c: len(_tokenize(c) & query_tokens), reverse=True)


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.
    """

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score an AI response using the judge LLM.
        """
        prompt = f"Question: {question}\nAnswer: {answer}\nRubric: {rubric}"
        llm_out = self.judge_llm_fn(prompt)
        try:
            import json
            parsed = json.loads(llm_out)
            if isinstance(parsed, dict):
                if "scores" in parsed and isinstance(parsed["scores"], dict):
                    scores = {k: float(v) for k, v in parsed["scores"].items()}
                    reasoning = str(parsed.get("reasoning", llm_out))
                else:
                    scores = {k: float(v) for k, v in parsed.items() if isinstance(v, (int, float))}
                    reasoning = llm_out
                return {"scores": scores, "reasoning": reasoning}
        except Exception:
            pass

        default_scores = {k: 0.5 for k in rubric.keys()}
        return {"scores": default_scores, "reasoning": llm_out}

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.
        """
        all_scores: list[float] = []
        for item in scores_batch:
            sc_dict = item.get("scores", {})
            all_scores.extend([float(v) for v in sc_dict.values() if isinstance(v, (int, float))])

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.5

        leniency_bias = avg_score > 0.8
        severity_bias = avg_score < 0.3

        positional_bias = False
        if len(scores_batch) >= 2:
            first_scores = [float(v) for v in scores_batch[0].get("scores", {}).values() if isinstance(v, (int, float))]
            first_avg = sum(first_scores) / len(first_scores) if first_scores else 0.0
            other_avgs = []
            for b in scores_batch[1:]:
                scs = [float(v) for v in b.get("scores", {}).values() if isinstance(v, (int, float))]
                if scs:
                    other_avgs.append(sum(scs) / len(scs))
            if other_avgs and first_avg > (sum(other_avgs) / len(other_avgs)) + 0.2:
                positional_bias = True

        return {
            "positional_bias": positional_bias,
            "leniency_bias": leniency_bias,
            "severity_bias": severity_bias,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """
        Run all QA pairs through the agent and evaluate each result.
        """
        results: list[EvalResult] = []
        for pair in qa_pairs:
            actual_answer = agent_fn(pair.question)
            ctxs = pair.retrieved_contexts
            if ctxs:
                ctxs = rerank_by_overlap(ctxs, pair.question)
            ctx_str = "\n".join(ctxs) if ctxs else (pair.context if pair.context is not None else "")
            res = evaluator.run_full_eval(
                answer=actual_answer,
                question=pair.question,
                context=ctx_str,
                expected=pair.expected_answer,
                contexts=ctxs if ctxs else None,
            )
            res.qa_pair = pair
            results.append(res)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """
        Generate an aggregate report from evaluation results.
        """
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "avg_completeness": 0.0,
                "avg_context_recall": None,
                "avg_context_precision": None,
                "failure_types": {},
            }

        passed_count = sum(1 for r in results if r.passed)
        pass_rate = passed_count / total
        avg_faithfulness = sum(r.faithfulness for r in results) / total
        avg_relevance = sum(r.relevance for r in results) / total
        avg_completeness = sum(r.completeness for r in results) / total

        recalls = [r.context_recall for r in results if r.context_recall is not None]
        avg_context_recall = (sum(recalls) / len(recalls)) if recalls else None

        precisions = [r.context_precision for r in results if r.context_precision is not None]
        avg_context_precision = (sum(precisions) / len(precisions)) if precisions else None

        failure_types: dict[str, int] = {}
        for r in results:
            if r.failure_type:
                failure_types[r.failure_type] = failure_types.get(r.failure_type, 0) + 1

        return {
            "total": total,
            "passed": passed_count,
            "pass_rate": pass_rate,
            "avg_faithfulness": avg_faithfulness,
            "avg_relevance": avg_relevance,
            "avg_completeness": avg_completeness,
            "avg_context_recall": avg_context_recall,
            "avg_context_precision": avg_context_precision,
            "failure_types": failure_types,
        }

    def run_regression(self, new_results: list[EvalResult], baseline_results: list[EvalResult]) -> dict[str, Any]:
        """
        Compare new evaluation results against a baseline.
        """
        new_total = len(new_results)
        base_total = len(baseline_results)

        new_f = sum(r.faithfulness for r in new_results) / new_total if new_total else 0.0
        new_r = sum(r.relevance for r in new_results) / new_total if new_total else 0.0
        new_c = sum(r.completeness for r in new_results) / new_total if new_total else 0.0

        base_f = sum(r.faithfulness for r in baseline_results) / base_total if base_total else 0.0
        base_r = sum(r.relevance for r in baseline_results) / base_total if base_total else 0.0
        base_c = sum(r.completeness for r in baseline_results) / base_total if base_total else 0.0

        regressions: list[str] = []
        if (base_f - new_f) > 0.05:
            regressions.append("faithfulness")
        if (base_r - new_r) > 0.05:
            regressions.append("relevance")
        if (base_c - new_c) > 0.05:
            regressions.append("completeness")

        return {
            "new_avg_faithfulness": new_f,
            "new_avg_relevance": new_r,
            "new_avg_completeness": new_c,
            "baseline_avg_faithfulness": base_f,
            "baseline_avg_relevance": base_r,
            "baseline_avg_completeness": base_c,
            "regressions": regressions,
            "passed": len(regressions) == 0,
        }

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """
        Return EvalResults where any score is below threshold.
        """
        failures: list[EvalResult] = []
        for r in results:
            if r.faithfulness < threshold or r.relevance < threshold or r.completeness < threshold:
                failures.append(r)
        return failures


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.
    """

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """
        Count failures by failure_type.
        """
        categories: dict[str, int] = {}
        for f in failures:
            ft = f.failure_type if f.failure_type else "unknown"
            categories[ft] = categories.get(ft, 0) + 1
        return categories

    def find_root_cause(self, failure: EvalResult) -> str:
        """
        Suggest a root cause for a single failure based on its scores.
        """
        f_score = failure.faithfulness
        r_score = failure.relevance
        c_score = failure.completeness

        scores = [("faithfulness", f_score), ("relevance", r_score), ("completeness", c_score)]
        min_val = min(s[1] for s in scores)
        lowest_metrics = [s[0] for s in scores if abs(s[1] - min_val) < 1e-6]

        if len(lowest_metrics) > 1 and min_val < 0.3:
            return "Multiple issues detected — review full pipeline"

        lowest = lowest_metrics[0]
        if lowest == "faithfulness":
            return "Context is missing or irrelevant — improve retrieval"
        elif lowest == "relevance":
            return "Answer does not address the question — improve prompt clarity"
        elif lowest == "completeness":
            return "Answer is missing key information — increase context window or improve generation"
        else:
            return "Multiple issues detected — review full pipeline"

    def generate_improvement_log(self, failures: list[EvalResult], suggestions: list[str]) -> str:
        """
        Generate a Markdown table logging failures and improvement actions.
        """
        lines = [
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |",
            "|------------|------|------------|---------------|--------|",
        ]
        for idx, f in enumerate(failures):
            fid = f"F{idx+1:03d}"
            ftype = f.failure_type if f.failure_type else "Unknown"
            root_cause = self.find_root_cause(f)
            fix = suggestions[idx] if idx < len(suggestions) else (suggestions[0] if suggestions else "Review pipeline")
            lines.append(f"| {fid} | {ftype} | {root_cause} | {fix} | Open |")
        return "\n".join(lines)

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """
        Generate a prioritized list of improvement suggestions based on failure patterns.
        """
        if not failures:
            return [
                "Maintain current retrieval and generation pipeline settings",
                "Expand test suite with additional adversarial edge cases",
                "Monitor real-time latency and production token consumption",
            ]

        cats = self.categorize_failures(failures)
        suggestions: list[str] = []
        if cats.get("hallucination", 0) > 0:
            suggestions.append("Implement hallucination checker to filter unsupported claims")
        if cats.get("irrelevant", 0) > 0 or cats.get("off_topic", 0) > 0:
            suggestions.append("Refine prompt clarity and intent classification to ensure direct answers")
        if cats.get("incomplete", 0) > 0:
            suggestions.append("Increase chunk size in RAG pipeline to reduce context fragmentation")
            suggestions.append("Add few-shot examples showing complete answers to improve completeness")

        defaults = [
            "Implement hallucination checker to filter unsupported claims",
            "Increase chunk size in RAG pipeline to reduce context fragmentation",
            "Add few-shot examples showing complete answers to improve completeness",
        ]
        for d in defaults:
            if d not in suggestions:
                suggestions.append(d)

        return suggestions


if __name__ == "__main__":
    qa_pairs = [
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
