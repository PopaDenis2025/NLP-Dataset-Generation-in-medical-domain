from typing import Optional, Literal, List
from pydantic import BaseModel, Field


class SourcePage(BaseModel):
    source_name: str
    url: str
    topic: str


class RawMedicalPage(BaseModel):
    source_name: str
    url: str
    topic: str
    title: Optional[str] = None
    raw_text: str
    status: Literal["ok", "failed", "too_short"]
    error: Optional[str] = None


class ExtractedMedicalRecord(BaseModel):
    symptom: str = Field(description="Medical symptom in English")
    disease: str = Field(description="Associated disease or condition in English")
    treatment_recommendation: str = Field(
        description="General educational treatment recommendation, not personal medical advice"
    )
    source_name: str
    source_url: str
    source_topic: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    model_used: str
    extraction_status: Literal["extracted", "needs_review", "rejected"] = "extracted"
    validation_status: Optional[Literal["valid", "needs_review", "rejected"]] = None
    validation_notes: Optional[str] = None


class ExtractedMedicalRecordList(BaseModel):
    records: List[ExtractedMedicalRecord]
