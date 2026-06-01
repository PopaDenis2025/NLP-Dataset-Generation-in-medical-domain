from crewai import Crew, Process

from agents import (
    create_medical_extractor_agent,
    create_medical_validator_agent,
    create_dataset_tester_agent,
)

from tasks import (
    create_extraction_task,
    create_validation_task,
    create_testing_task,
)


def create_medical_crew():
    extractor = create_medical_extractor_agent()
    validator = create_medical_validator_agent()
    tester = create_dataset_tester_agent()

    extraction_task = create_extraction_task(extractor)
    validation_task = create_validation_task(validator)
    testing_task = create_testing_task(tester)

    return Crew(
        agents=[extractor, validator, tester],
        tasks=[extraction_task, validation_task, testing_task],
        process=Process.sequential,
        verbose=True,
    )