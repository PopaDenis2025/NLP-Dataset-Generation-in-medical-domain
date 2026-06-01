import argparse

from nlp_medical_dataset.pipelines.dataset_pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate medical NLP dataset using CrewAI and Ollama.")
    parser.add_argument("--skip-scrape", action="store_true", help="Use existing data/raw/pages_raw.jsonl.")
    args = parser.parse_args()

    run_pipeline(skip_scrape=args.skip_scrape)


if __name__ == "__main__":
    main()
