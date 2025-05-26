import asyncio
import json
from pathlib import Path

from src.evaluation.evaluator import ModelEvaluator
from src.models.model_factory import ModelFactory


async def main():
    # Load sample questions
    questions_path = Path("data/questions/sample_questions.json")
    with open(questions_path) as f:
        questions = json.load(f)["questions"]

    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Create models
    models = {
        "OpenAI GPT-4": ModelFactory.create_model("openai", "gpt-4"),
        "OpenAI GPT-3.5": ModelFactory.create_model("openai", "gpt-3.5-turbo"),
        "Gemini Pro": ModelFactory.create_model("gemini"),
        "Local LLaMA2": ModelFactory.create_model("ollama", "llama2"),
        "Local Mixtral": ModelFactory.create_model("ollama", "mixtral"),
    }

    # Evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        try:
            result = await evaluator.evaluate_model(model, questions)
            results[name] = result
            print(f"Completed evaluation for {name}")
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")

    # Compare results
    comparison_path = evaluator.compare_models(results)
    print(f"\nResults saved to: {comparison_path}")

    # Clean up
    for model in models.values():
        if hasattr(model, "close"):
            await model.close()


if __name__ == "__main__":
    asyncio.run(main())
