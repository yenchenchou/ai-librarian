import asyncio
import json
from pathlib import Path

from src.analysis.question_classifier import QuestionClassifier
from src.scrapers.libanswers_scraper import LibAnswersScraper


async def main():
    # Initialize scraper and classifier
    scraper = LibAnswersScraper()
    classifier = QuestionClassifier()

    try:
        # Scrape questions
        print("Scraping questions from UWM LibAnswers...")
        questions = await scraper.scrape_questions(max_pages=10)

        # Save raw questions
        questions_path = scraper.save_questions(questions)
        print(f"Saved {len(questions)} questions to {questions_path}")

        # Classify questions
        print("\nClassifying questions...")
        results = classifier.classify_questions(questions)

        # Save classification results
        results_path = classifier.save_classification_results(results)
        print(f"Saved classification results to {results_path}")

        # Print summary
        print("\nClassification Summary:")
        print(f"Total questions analyzed: {results['statistics']['total_questions']}")
        print("\nCategory Distribution:")
        for category, stats in results["statistics"]["category_distribution"].items():
            print(
                f"- {category}: {stats['count']} questions ({stats['percentage']:.1f}%)"
            )

    finally:
        # Clean up
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())
