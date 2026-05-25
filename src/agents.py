import os
from dotenv import load_dotenv
from crewai import Agent, LLM

load_dotenv()


def create_local_llm() -> LLM:
    return LLM(
        model=os.getenv("OPENAI_MODEL", "ollama/llama3.2:3b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        temperature=0.1,
    )


def create_medical_extractor_agent() -> Agent:
    return Agent(
        role="Medical NLP Data Extractor",
        goal=(
            "Extract symptom-disease-treatment records "
            "from trusted English medical text."
        ),
        backstory=(
            "You are a medical NLP extraction agent. "
            "You extract only facts supported by the input text. "
            "You return strict JSON only."
        ),
        llm=create_local_llm(),
        verbose=False,
        allow_delegation=False,
    )


def create_medical_validator_agent() -> Agent:
    return Agent(
        role="Medical Dataset Validator and Cleaner",
        goal=(
            "Validate, clean, normalize and filter extracted medical "
            "dataset records."
        ),
        backstory=(
            "You are a careful medical dataset quality agent. "
            "You check whether each record is safe, clear, useful, "
            "and suitable for an educational NLP dataset."
        ),
        llm=create_local_llm(),
        verbose=False,
        allow_delegation=False,
    )


def create_dataset_tester_agent() -> Agent:
    return Agent(
        role="Medical Dataset Quality Tester",
        goal=(
            "Analyze the quality of a generated medical NLP dataset "
            "using clear metrics and report its strengths, weaknesses, "
            "risks, and improvement steps."
        ),
        backstory=(
            "You are an intelligent systems evaluator. "
            "You analyze datasets, detect quality problems, "
            "and write concise technical reports for academic projects."
        ),
        llm=create_local_llm(),
        verbose=False,
        allow_delegation=False,
    )