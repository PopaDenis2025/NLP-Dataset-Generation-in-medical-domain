from crewai import Crew, Process

from nlp_medical_dataset.agents.medical_agents import (
    create_dataset_tester_agent,
    create_medical_extractor_agent,
    create_medical_validator_agent,
)


def create_medical_dataset_crew() -> Crew:
    return Crew(
        agents=[
            create_medical_extractor_agent(),
            create_medical_validator_agent(),
            create_dataset_tester_agent(),
        ],
        tasks=[],
        process=Process.sequential,
        verbose=False,
    )
