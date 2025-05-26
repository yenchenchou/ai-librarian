import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class ModelEvaluator:
    """Framework for evaluating AI models on library-related tasks."""

    def __init__(self, output_dir: str = "data/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def evaluate_model(
        self, model, questions: List[Dict[str, str]], metrics: List[str] = None
    ) -> Dict:
        """
        Evaluate a model on a set of questions.

        Args:
            model: An instance of BaseAIModel
            questions: List of questions with metadata
            metrics: List of metrics to evaluate (e.g., ['accuracy', 'relevance', 'cost'])
        """
        results = []

        for question in questions:
            response = await model.generate_response(
                prompt=question["text"], context=question.get("context", None)
            )

            # Calculate metrics
            metrics_results = self._calculate_metrics(
                question=question, response=response, model=model
            )

            results.append(
                {
                    "question_id": question["id"],
                    "question": question["text"],
                    "response": response,
                    "metrics": metrics_results,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Save results
        self._save_results(results, model.model_name)

        return self._aggregate_results(results)

    def _calculate_metrics(
        self, question: Dict, response: str, model
    ) -> Dict[str, float]:
        """Calculate evaluation metrics for a single question-response pair."""
        # TODO: Implement specific metrics calculation
        return {
            "relevance": 0.0,
            "accuracy": 0.0,
            "cost": model.get_cost_estimate(len(question["text"]), len(response)),
        }

    def _save_results(self, results: List[Dict], model_name: str):
        """Save evaluation results to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{model_name}_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results across all questions."""
        metrics_df = pd.DataFrame([r["metrics"] for r in results])

        return {
            "mean_metrics": metrics_df.mean().to_dict(),
            "std_metrics": metrics_df.std().to_dict(),
            "total_questions": len(results),
            "total_cost": sum(r["metrics"]["cost"] for r in results),
        }

    def compare_models(
        self, model_results: Dict[str, Dict], output_format: str = "json"
    ) -> str:
        """
        Compare results from multiple models.

        Args:
            model_results: Dictionary of model names to their evaluation results
            output_format: Output format ('json' or 'csv')
        """
        comparison = pd.DataFrame(model_results).T

        if output_format == "csv":
            output_path = (
                self.output_dir
                / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            comparison.to_csv(output_path)
        else:
            output_path = (
                self.output_dir
                / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            comparison.to_json(output_path, indent=2)

        return str(output_path)
