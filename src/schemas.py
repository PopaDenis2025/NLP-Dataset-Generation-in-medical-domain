from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, Literal, List


class SourcePage(BaseModel):
    source_name: str
    url: HttpUrl
    topic: str


class RawMedicalPage(BaseModel):
    source_name: str
    url: str
    topic: str
    title: Optional[str] = None
    raw_text: str
    status: str
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
    evidence: str = Field(description="Short evidence from source text")
    model_used: str
    extraction_status: Literal["extracted", "needs_review", "rejected"]
    notes: Optional[str] = None


class ExtractedMedicalRecordList(BaseModel):
    records: List[ExtractedMedicalRecord]