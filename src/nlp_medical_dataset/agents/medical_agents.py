from crewai import Agent, LLM
from nlp_medical_dataset.config.settings import OLLAMA_MODEL


def _llm() -> LLM:
    return LLM(model=OLLAMA_MODEL, temperature=0.1)


def create_medical_extractor_agent() -> Agent:
    return Agent(
        role="Medical Data Extractor",
        goal="Extract clean symptom-disease-treatment records from trusted medical text.",
        backstory=(
            "You are an NLP dataset engineer specialized in medical text. "
            "You extract structured rows for educational datasets only."
        ),
        llm=_llm(),
        verbose=False,
        allow_delegation=False,
    )


def create_medical_validator_agent() -> Agent:
    return Agent(
        role="Medical Dataset Validator",
        goal="Validate and clean medical dataset rows, rejecting unsafe or unclear records.",
        backstory=(
            "You are a careful medical NLP data reviewer. "
            "You do not provide personal medical advice and you reject risky rows."
        ),
        llm=_llm(),
        verbose=False,
        allow_delegation=False,
    )


def create_dataset_tester_agent() -> Agent:
    return Agent(
        role="Dataset Quality Analyst",
        goal="Evaluate dataset quality and write concise technical reports.",
        backstory="You analyze NLP datasets, data risks, missing fields, duplicates and class balance.",
        llm=_llm(),
        verbose=False,
        allow_delegation=False,
    )
