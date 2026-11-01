from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator

N_FEATURES = 30


class PredictRequest(BaseModel):
    """A single sample: the 30 numeric features, in the order returned by GET /model-info."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [
                    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
                ]
            }
        }
    )

    features: List[float] = Field(..., description=f"Exactly {N_FEATURES} numeric feature values.")

    @field_validator("features")
    @classmethod
    def check_features(cls, v: List[float]) -> List[float]:
        if len(v) != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} features, got {len(v)}")
        if any(x != x or x in (float("inf"), float("-inf")) for x in v):
            raise ValueError("features must be finite numbers")
        return v


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class id (0 = malignant, 1 = benign).")
    label: str = Field(..., description="Human-readable class name.")
    confidence: float = Field(..., ge=0, le=1, description="Probability of the predicted class.")
    probabilities: Dict[str, float] = Field(..., description="Probability for every class, keyed by label.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfo(BaseModel):
    model_type: str
    n_features: int
    feature_names: List[str]
    classes: Dict[str, str]
    cv_roc_auc: float | None = None
    test_roc_auc: float | None = None
