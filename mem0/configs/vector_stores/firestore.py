import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class FirestoreConfig(BaseModel):
    """Configuration for Firestore vector database."""

    collection_name: str = Field("mem0_vectors", description="Name of the Firestore collection for vectors")
    embedding_model_dims: Optional[int] = Field(None, description="Dimensions of the embedding model. Required if create_col is to be called from __init__.")
    service_account_key_path: Optional[str] = Field(None, description="Path to the Firebase service account key JSON file. If None, uses GOOGLE_APPLICATION_CREDENTIALS.")
    vector_field_name: str = Field("vector", description="Name of the field in Firestore documents that stores the vector embeddings.")
    metric: str = Field("cosine", description="Distance metric for vector similarity. Supported: 'cosine', 'euclidean', 'dotproduct'.")
    batch_size: int = Field(100, description="Batch size for write operations (insert, delete_col). Min 1.")
    # Add any other Firestore-specific parameters you might foresee, e.g., project_id if not covered by service account.

    @model_validator(mode="before")
    @classmethod
    def check_service_account_path_or_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        service_account_key_path = values.get("service_account_key_path")
        if not service_account_key_path and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            # This is more of a runtime check for the DB class, but good to note.
            # Pydantic validation happens at config load, SDK init happens later.
            # We might not want to hard fail here if GOOGLE_APPLICATION_CREDENTIALS might be set in the execution environment.
            # For now, let's just ensure the logic is sound for the DB class.
            # Consider if this validation is strictly needed at config parsing vs. DB initialization.
            # If service_account_key_path is None, the FirestoreDB class will attempt to use env var.
            pass # No explicit error here, handled by SDK/DB class.
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_metric(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        metric = values.get("metric")
        if metric and metric.lower() not in ["cosine", "euclidean", "dotproduct"]:
            raise ValueError(f"Invalid metric: '{metric}'. Supported metrics are 'cosine', 'euclidean', 'dotproduct'.")
        if metric: # Ensure it's stored in lowercase
            values["metric"] = metric.lower()
        return values
    
    @model_validator(mode="before")
    @classmethod
    def validate_batch_size(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        batch_size = values.get("batch_size")
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"'batch_size' must be at least 1. Got {batch_size}")
        return values

    # Placeholder for future validation of extra fields if strictness is desired,
    # similar to PineconeConfig. For now, allowing extra fields.
    # @model_validator(mode="before")
    # @classmethod
    # def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    #     allowed_fields = set(cls.model_fields.keys())
    #     input_fields = set(values.keys())
    #     extra_fields = input_fields - allowed_fields
    #     if extra_fields:
    #         raise ValueError(
    #             f"Extra fields not allowed: {', '.join(extra_fields)}. "
    #             f"Please input only the following fields: {', '.join(allowed_fields)}"
    #         )
    #     return values

    model_config = {
        "arbitrary_types_allowed": True, # As Firestore client might be passed or other complex types if extended.
        # "extra":"forbid" # If you want to strictly forbid extra fields not defined.
    } 