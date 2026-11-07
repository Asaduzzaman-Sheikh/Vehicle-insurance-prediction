""" 
    This artifact entity is used to define the structure of the artifacts generated during the training pipeline.
    It contains the paths to the directories and files where the artifacts will be stored.
"""

from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """A data class to represent the artifacts generated during data ingestion."""
    train_file_path: str
    test_file_path: str
    
    
@dataclass
class DataValidationArtifact:
    """A data class to represent the artifacts generated during data validation."""
    validation_status: bool
    error_message: str
    validation_report_file_path: str
@dataclass
class DataTransformationArtifact:
    """A data class to represent the artifacts generated during data transformation."""
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str
    

@dataclass
class ClassificationMetricArtifact:
    """A data class to represent the classification metrics generated during model evaluation."""
    accuracy: float
    f1_score: float
    precision_score: float
    recall_score: float
@dataclass
class ModelTrainerArtifact:
    """A data class to represent the artifacts generated during model training."""
    trained_model_file_path: str
    metrics_artifact: ClassificationMetricArtifact
    