from crew import create_medical_crew


def main():
    crew = create_medical_crew()

    result = crew.kickoff(inputs={
        "text": "Patient has fever and cough. Possible flu. Treatment includes rest and fluids."
    })

    print(result)


if __name__ == "__main__":
    main()
