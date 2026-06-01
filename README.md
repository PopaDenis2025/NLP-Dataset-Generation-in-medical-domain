# NLP Dataset Generation in Medical Domain

## Run

```bash
ollama pull mistral
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m nlp_medical_dataset.main
```

Final dataset:
- `data/final/medical_dataset.csv`
- `data/final/medical_dataset.jsonl`

Reports:
- `reports/dataset_metrics.json`
- `reports/dataset_quality_report.md`
