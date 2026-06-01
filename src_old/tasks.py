from crewai import Task


def create_extraction_task(agent) -> Task:
    return Task(
        description=(
            "Extract symptom-disease-treatment records from this text:\n\n"
            "{text}\n\n"
            "Return strict JSON only. "
            "Each record must contain: symptom, disease, treatment, evidence."
        ),
        expected_output=(
            "Strict JSON array. Each object has: "
            "symptom, disease, treatment, evidence."
        ),
        agent=agent,
    )


def create_validation_task(agent) -> Task:
    return Task(
        description=(
            "Validate and clean the extracted medical records. "
            "Remove unsupported, unclear, duplicated or unsafe records. "
            "Normalize symptoms, diseases and treatments."
        ),
        expected_output=(
            "Strict JSON array with only valid cleaned records."
        ),
        agent=agent,
    )


def create_testing_task(agent) -> Task:
    return Task(
        description=(
            "Analyze the final dataset quality. "
            "Report number of records, strengths, weaknesses, risks, "
            "and improvement steps."
        ),
        expected_output=(
            "A concise technical quality report."
        ),
        agent=agent,
    )