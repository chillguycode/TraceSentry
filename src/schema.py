from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, AliasChoices, ConfigDict, field_validator


# Raw Ingestion Model (with resilient type validators)
class SwarmTracesRawRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str = Field(..., validation_alias=AliasChoices("id", "row_id", "event_id"))
    cite: Optional[str] = None
    kind: str = Field(default="payload", validation_alias=AliasChoices("kind", "record_type", "type"))
    parent_id: Optional[str] = Field(default=None, validation_alias=AliasChoices("parent_id", "parent"))
    depth: int = 0
    children: int = 0
    source_rows: List[str] = Field(default_factory=list)
    time_utc: Optional[str] = None
    text: str = Field(default="", validation_alias=AliasChoices("text", "payload", "recovered_text", "content"))
    tags: List[str] = Field(default_factory=list)

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, v: Any) -> List[str]:
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        if isinstance(v, list):
            return v
        return []

    @field_validator("source_rows", mode="before")
    @classmethod
    def normalize_source_rows(cls, v: Any) -> List[str]:
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return v
        return []


# Extracted Malicious Payload Artifact
class ExtractedAttackArtifact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    destination_url: Optional[str] = None
    file_path: Optional[str] = None
    injected_commands: List[str] = Field(default_factory=list)
    submitted_code: Optional[str] = None
    has_command_injection: bool = False


# Normalized Graph Event
class NormalizedForensicEvent(BaseModel):
    step: int
    event_id: str
    timestamp: str = "undated"
    sender_id: str
    receiver_id: str
    action_type: str
    artifact: ExtractedAttackArtifact = Field(default_factory=ExtractedAttackArtifact)
    raw_content: str = ""
