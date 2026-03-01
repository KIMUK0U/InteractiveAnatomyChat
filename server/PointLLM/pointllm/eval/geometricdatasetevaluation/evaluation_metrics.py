"""
Evaluation Metrics for Multi-Stage Chain-of-Thought Inference Results

This module provides evaluation capabilities for:
1. CoT quality (coherence, relevance)
2. Final answer accuracy (exact match, partial match, semantic similarity)
3. Comprehensive reporting
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re


# ============================================================================
# Evaluation Metrics
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.rstrip('.')
    return text


def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """Calculate exact match score (0.0 or 1.0)."""
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)
    return 1.0 if pred_norm == gt_norm else 0.0


def calculate_partial_match(prediction: str, ground_truth: str) -> float:
    """Calculate token-level F1 score."""
    pred_tokens = set(normalize_text(prediction).split())
    gt_tokens = set(normalize_text(ground_truth).split())
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    common = pred_tokens.intersection(gt_tokens)
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    num_scenes: int
    num_qa_pairs: int
    avg_exact_match: float
    avg_partial_match: float
    avg_semantic_similarity: Optional[float] = None
    by_qa_type: Optional[Dict] = None
    cot_metrics: Optional[Dict] = None


# ============================================================================
# CoT Quality Evaluation (Optional - requires LLM)
# ============================================================================

class CoTQualityEvaluator:
    """
    Evaluate the quality of Chain of Thought steps.
    
    This is optional and requires an LLM API (OpenAI or Ollama).
    """
    
    def __init__(self, use_ollama: bool = False, model: str = "qwen3:8b"):
        self.use_ollama = use_ollama
        self.model = model
        
        if use_ollama:
            try:
                import ollama
                self.client = ollama
            except ImportError:
                print("[Warning] Ollama not installed. CoT quality evaluation disabled.")
                self.client = None
        else:
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("[Warning] OPENAI_API_KEY not found. CoT quality evaluation disabled.")
                    self.client = None
                else:
                    self.client = OpenAI(api_key=api_key)
            except ImportError:
                print("[Warning] OpenAI not installed. CoT quality evaluation disabled.")
                self.client = None
    
    def evaluate_cot_quality(self, cot_steps: str, question: str) -> Dict:
        """
        Evaluate CoT quality on three criteria:
        - Coherence: Are steps logical and well-structured?
        - Relevance: Are steps relevant to the question?
        - Completeness: Do steps cover all necessary reasoning?
        
        Returns scores 0.0-1.0 for each criterion.
        """
        if not self.client:
            return {"coherence": None, "relevance": None, "completeness": None}
        
        prompt = f"""Evaluate the quality of the following Chain of Thought (CoT) reasoning steps for the given question.

Question: {question}

CoT Steps:
{cot_steps}

Rate the CoT on three criteria (0.0 to 1.0 scale):
1. Coherence: Are the steps logical and well-structured?
2. Relevance: Are the steps directly relevant to answering the question?
3. Completeness: Do the steps cover all necessary reasoning to answer the question?

Respond in JSON format:
{{
  "coherence": 0.0-1.0,
  "relevance": 0.0-1.0,
  "completeness": 0.0-1.0,
  "explanation": "brief explanation"
}}
"""
        
        try:
            if self.use_ollama:
                response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response['message']['content']
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                content = response.choices[0].message.content
            
            # Parse JSON response
            import json
            result = json.loads(content)
            return {
                "coherence": float(result.get("coherence", 0.5)),
                "relevance": float(result.get("relevance", 0.5)),
                "completeness": float(result.get("completeness", 0.5)),
                "explanation": result.get("explanation", "")
            }
        
        except Exception as e:
            print(f"[Warning] CoT quality evaluation failed: {e}")
            return {"coherence": None, "relevance": None, "completeness": None}


# ============================================================================
# Semantic Similarity Evaluator (from original code)
# ============================================================================

class SemanticSimilarityEvaluator:
    """Evaluate semantic similarity using LLM."""
    
    def __init__(self, use_ollama: bool = False, model: str = "qwen3:8b"):
        self.use_ollama = use_ollama
        self.model = model
        
        if use_ollama:
            try:
                import ollama
                self.client = ollama
            except ImportError:
                print("[Warning] Ollama not installed. Semantic evaluation disabled.")
                self.client = None
        else:
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("[Warning] OPENAI_API_KEY not found. Semantic evaluation disabled.")
                    self.client = None
                else:
                    self.client = OpenAI(api_key=api_key)
            except ImportError:
                print("[Warning] OpenAI not installed. Semantic evaluation disabled.")
                self.client = None
    
    def evaluate_semantic_similarity(self, prediction: str, ground_truth: str, 
                                    question: str = "") -> float:
        """
        Evaluate semantic similarity (0.0 to 1.0).
        """
        if not self.client:
            return None
        
        prompt = f"""Compare the semantic similarity between the predicted answer and the ground truth answer.

Question: {question}

Ground Truth: {ground_truth}

Predicted Answer: {prediction}

Rate the semantic similarity on a scale of 0.0 to 1.0:
- 1.0: Identical meaning
- 0.8-0.9: Minor differences but essentially correct
- 0.5-0.7: Partially correct
- 0.2-0.4: Mostly incorrect but some relevant elements
- 0.0-0.1: Completely incorrect

Respond with ONLY a number between 0.0 and 1.0.
"""
        
        try:
            if self.use_ollama:
                response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response['message']['content'].strip()
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                content = response.choices[0].message.content.strip()
            
            # Extract number
            match = re.search(r'(\d+\.?\d*)', content)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)
            return 0.5
        
        except Exception as e:
            print(f"[Warning] Semantic evaluation failed: {e}")
            return None


# ============================================================================
# Main Evaluator for CoT Inference Results
# ============================================================================

class CoTInferenceEvaluator:
    """
    Comprehensive evaluator for multi-stage CoT inference results.
    """
    
    def __init__(self, 
                 use_semantic: bool = False,
                 use_cot_quality: bool = False,
                 use_ollama: bool = False,
                 model: str = "qwen3:8b"):
        """
        Args:
            use_semantic: Enable semantic similarity evaluation
            use_cot_quality: Enable CoT quality evaluation
            use_ollama: Use Ollama instead of OpenAI
            model: Model name for LLM evaluations
        """
        self.use_semantic = use_semantic
        self.use_cot_quality = use_cot_quality
        
        self.semantic_evaluator = None
        self.cot_evaluator = None
        
        if use_semantic:
            self.semantic_evaluator = SemanticSimilarityEvaluator(
                use_ollama=use_ollama,
                model=model
            )
        
        if use_cot_quality:
            self.cot_evaluator = CoTQualityEvaluator(
                use_ollama=use_ollama,
                model=model
            )
    
    def evaluate_csv(self, csv_path: str) -> EvaluationResult:
        """
        Evaluate a CSV file containing CoT inference results.
        
        Expected columns:
        - scene_id
        - qa_index
        - qa_type
        - question
        - ground_truth
        - cot_steps
        - cot_answers
        - predicted_answer
        """
        df = pd.read_csv(csv_path)
        
        # Basic metrics
        exact_matches = []
        partial_matches = []
        semantic_scores = []
        
        # CoT quality metrics
        cot_coherence = []
        cot_relevance = []
        cot_completeness = []
        
        # By QA type
        by_qa_type = {}
        
        for idx, row in df.iterrows():
            question = row['question']
            ground_truth = row['ground_truth']
            predicted = row['predicted_answer']
            qa_type = row.get('qa_type', 'unknown')
            cot_steps = row.get('cot_steps', '')
            
            # Basic metrics
            em = calculate_exact_match(predicted, ground_truth)
            pm = calculate_partial_match(predicted, ground_truth)
            
            exact_matches.append(em)
            partial_matches.append(pm)
            
            # Semantic similarity
            if self.use_semantic and self.semantic_evaluator:
                sem = self.semantic_evaluator.evaluate_semantic_similarity(
                    predicted, ground_truth, question
                )
                if sem is not None:
                    semantic_scores.append(sem)
            
            # CoT quality
            if self.use_cot_quality and self.cot_evaluator and cot_steps:
                cot_quality = self.cot_evaluator.evaluate_cot_quality(cot_steps, question)
                if cot_quality['coherence'] is not None:
                    cot_coherence.append(cot_quality['coherence'])
                if cot_quality['relevance'] is not None:
                    cot_relevance.append(cot_quality['relevance'])
                if cot_quality['completeness'] is not None:
                    cot_completeness.append(cot_quality['completeness'])
            
            # By QA type
            if qa_type not in by_qa_type:
                by_qa_type[qa_type] = {'em': [], 'pm': [], 'sem': []}
            by_qa_type[qa_type]['em'].append(em)
            by_qa_type[qa_type]['pm'].append(pm)
            if semantic_scores:
                by_qa_type[qa_type]['sem'].append(semantic_scores[-1])
        
        # Aggregate results
        num_scenes = df['scene_id'].nunique()
        num_qa_pairs = len(df)
        
        avg_exact_match = np.mean(exact_matches) if exact_matches else 0.0
        avg_partial_match = np.mean(partial_matches) if partial_matches else 0.0
        avg_semantic = np.mean(semantic_scores) if semantic_scores else None
        
        # CoT metrics
        cot_metrics = None
        if cot_coherence or cot_relevance or cot_completeness:
            cot_metrics = {
                "avg_coherence": np.mean(cot_coherence) if cot_coherence else None,
                "avg_relevance": np.mean(cot_relevance) if cot_relevance else None,
                "avg_completeness": np.mean(cot_completeness) if cot_completeness else None
            }
        
        # By QA type summary
        by_qa_type_summary = {}
        for qa_type, metrics in by_qa_type.items():
            by_qa_type_summary[qa_type] = {
                "count": len(metrics['em']),
                "avg_em": np.mean(metrics['em']),
                "avg_pm": np.mean(metrics['pm']),
                "avg_sem": np.mean(metrics['sem']) if metrics['sem'] else None
            }
        
        return EvaluationResult(
            num_scenes=num_scenes,
            num_qa_pairs=num_qa_pairs,
            avg_exact_match=avg_exact_match,
            avg_partial_match=avg_partial_match,
            avg_semantic_similarity=avg_semantic,
            by_qa_type=by_qa_type_summary,
            cot_metrics=cot_metrics
        )


# ============================================================================
# Standalone Evaluation Function
# ============================================================================

def run_standalone_evaluation(base_dir: str,
                              use_semantic: bool = False,
                              use_cot_quality: bool = False,
                              use_ollama: bool = False,
                              model: str = "qwen3:8b") -> Dict[str, str]:
    """
    Run evaluation on all inference_results_cot.csv files in base_dir.
    
    Returns:
        Dictionary of output file paths
    """
    evaluator = CoTInferenceEvaluator(
        use_semantic=use_semantic,
        use_cot_quality=use_cot_quality,
        use_ollama=use_ollama,
        model=model
    )
    
    base_path = Path(base_dir)
    csv_files = list(base_path.rglob("results/inference_results_cot.csv"))
    
    print(f"\nFound {len(csv_files)} inference result files")
    
    all_results = []
    
    for csv_path in csv_files:
        folder_name = csv_path.parent.parent.name
        print(f"\nEvaluating: {folder_name}")
        
        result = evaluator.evaluate_csv(str(csv_path))
        
        print(f"  Scenes: {result.num_scenes}")
        print(f"  QA Pairs: {result.num_qa_pairs}")
        print(f"  Exact Match: {result.avg_exact_match:.4f}")
        print(f"  Partial Match: {result.avg_partial_match:.4f}")
        if result.avg_semantic_similarity:
            print(f"  Semantic Similarity: {result.avg_semantic_similarity:.4f}")
        if result.cot_metrics:
            print(f"  CoT Coherence: {result.cot_metrics['avg_coherence']:.4f}")
            print(f"  CoT Relevance: {result.cot_metrics['avg_relevance']:.4f}")
            print(f"  CoT Completeness: {result.cot_metrics['avg_completeness']:.4f}")
        
        all_results.append({
            "folder": folder_name,
            "num_scenes": result.num_scenes,
            "num_qa_pairs": result.num_qa_pairs,
            "exact_match": result.avg_exact_match,
            "partial_match": result.avg_partial_match,
            "semantic_similarity": result.avg_semantic_similarity,
            "cot_coherence": result.cot_metrics['avg_coherence'] if result.cot_metrics else None,
            "cot_relevance": result.cot_metrics['avg_relevance'] if result.cot_metrics else None,
            "cot_completeness": result.cot_metrics['avg_completeness'] if result.cot_metrics else None
        })
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = base_path / f"evaluation_summary_cot_{timestamp}.csv"
    
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Evaluation summary saved to: {summary_path}")
    print(f"{'='*60}")
    
    return {"summary": str(summary_path)}


# ============================================================================
# Main Function (for standalone execution)
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CoT Inference Results")
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--use_semantic", action="store_true")
    parser.add_argument("--use_cot_quality", action="store_true")
    parser.add_argument("--use_ollama", action="store_true")
    parser.add_argument("--model", type=str, default="qwen3:8b")
    
    args = parser.parse_args()
    
    run_standalone_evaluation(
        base_dir=args.base_dir,
        use_semantic=args.use_semantic,
        use_cot_quality=args.use_cot_quality,
        use_ollama=args.use_ollama,
        model=args.model
    )


if __name__ == "__main__":
    main()