"""Validated HTTP request and response contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RelevantExperience(StrEnum):
    """Allowed relevant-experience values after source normalization."""

    HAS_RELEVANT_EXPERIENCE = "Has relevant experience"
    NO_RELEVANT_EXPERIENCE = "No relevant experience"


class EnrolledUniversity(StrEnum):
    """Allowed university-enrollment values."""

    FULL_TIME_COURSE = "Full time course"
    PART_TIME_COURSE = "Part time course"
    NO_ENROLLMENT = "no_enrollment"


class EducationLevel(StrEnum):
    """Allowed education levels."""

    PRIMARY_SCHOOL = "Primary School"
    HIGH_SCHOOL = "High School"
    GRADUATE = "Graduate"
    MASTERS = "Masters"
    PHD = "Phd"


class MajorDiscipline(StrEnum):
    """Allowed major-discipline values."""

    ARTS = "Arts"
    BUSINESS_DEGREE = "Business Degree"
    HUMANITIES = "Humanities"
    NO_MAJOR = "No Major"
    OTHER = "Other"
    STEM = "STEM"


class Experience(StrEnum):
    """Allowed years-of-experience bands."""

    LESS_THAN_ONE = "<1"
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    ELEVEN = "11"
    TWELVE = "12"
    THIRTEEN = "13"
    FOURTEEN = "14"
    FIFTEEN = "15"
    SIXTEEN = "16"
    SEVENTEEN = "17"
    EIGHTEEN = "18"
    NINETEEN = "19"
    TWENTY = "20"
    MORE_THAN_TWENTY = ">20"


class CompanySize(StrEnum):
    """Allowed company-size bands."""

    LESS_THAN_TEN = "<10"
    TEN_TO_FORTY_NINE = "10/49"
    FIFTY_TO_NINETY_NINE = "50-99"
    ONE_HUNDRED_TO_FIVE_HUNDRED = "100-500"
    FIVE_HUNDRED_TO_NINE_HUNDRED_NINETY_NINE = "500-999"
    ONE_THOUSAND_TO_FOUR_THOUSAND_NINE_HUNDRED_NINETY_NINE = "1000-4999"
    FIVE_THOUSAND_TO_NINE_THOUSAND_NINE_HUNDRED_NINETY_NINE = "5000-9999"
    TEN_THOUSAND_OR_MORE = "10000+"


class CompanyType(StrEnum):
    """Allowed company-type values."""

    EARLY_STAGE_STARTUP = "Early Stage Startup"
    FUNDED_STARTUP = "Funded Startup"
    NGO = "NGO"
    OTHER = "Other"
    PUBLIC_SECTOR = "Public Sector"
    PRIVATE_LIMITED = "Pvt Ltd"


class LastNewJob(StrEnum):
    """Allowed years-since-last-job-change bands."""

    NEVER = "never"
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    MORE_THAN_FOUR = ">4"


class CandidateFeatures(BaseModel):
    """Features available before a participant begins training."""

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        json_schema_extra={
            "examples": [
                {
                    "city_development_index": 0.92,
                    "city": "city_103",
                    "relevant_experience": "Has relevant experience",
                    "enrolled_university": "no_enrollment",
                    "education_level": "Graduate",
                    "major_discipline": "STEM",
                    "experience": "10",
                    "company_size": "100-500",
                    "company_type": "Pvt Ltd",
                    "last_new_job": "1",
                }
            ]
        },
    )

    city_development_index: Annotated[
        float,
        Field(strict=True, ge=0, le=1, allow_inf_nan=False),
    ]
    city: Annotated[
        str,
        Field(
            min_length=6,
            max_length=20,
            pattern=r"^city_[0-9]+$",
            description="An anonymized city identifier such as city_103.",
        ),
    ]
    relevant_experience: RelevantExperience
    enrolled_university: EnrolledUniversity | None
    education_level: EducationLevel | None
    major_discipline: MajorDiscipline | None
    experience: Experience | None
    company_size: CompanySize | None
    company_type: CompanyType | None
    last_new_job: LastNewJob | None

    @field_validator(
        "city",
        "relevant_experience",
        "enrolled_university",
        "education_level",
        "major_discipline",
        "experience",
        "company_size",
        "company_type",
        "last_new_job",
        mode="before",
    )
    @classmethod
    def normalize_category(cls, value: object) -> object:
        """Strip category whitespace and normalize blanks to missing values."""
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class BatchPredictionRequest(BaseModel):
    """Bounded batch inference request."""

    model_config = ConfigDict(extra="forbid")

    candidates: Annotated[
        list[CandidateFeatures],
        Field(min_length=1, max_length=1_000),
    ]


class PredictionResponse(BaseModel):
    """Risk probabilities for one participant."""

    job_change_intent_probability: Annotated[float, Field(ge=0, le=1)]
    no_job_change_intent_probability: Annotated[float, Field(ge=0, le=1)]


class BatchPredictionItem(PredictionResponse):
    """One batch result with stable input identity and business ranking."""

    input_index: Annotated[int, Field(ge=0)]
    training_priority_rank: Annotated[int, Field(ge=1)]


class BatchPredictionResponse(BaseModel):
    """Ranked batch results."""

    predictions: list[BatchPredictionItem]


class HealthResponse(BaseModel):
    """Readiness response."""

    status: str
    model_uri: str


class ModelInfoResponse(BaseModel):
    """Reader-facing model lineage and serving contract."""

    model_uri: str
    model_name: str | None
    model_version: str | None
    model_alias: str | None
    run_id: str | None
    git_commit: str | None
    source_sha256: str | None
    optuna_study: str | None
    optuna_trial: int | None
    preprocessing: str | None
    parameters: dict[str, str]
    test_metrics: dict[str, float]
    features: list[str]
    prediction_output: str
