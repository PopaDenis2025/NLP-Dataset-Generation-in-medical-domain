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
        verbose=True,
        allow_delegation=False,
    )