""" 
    This artifact entity is used to define the structure of the artifacts generated during the training pipeline.
    It contains the paths to the directories and files where the artifacts will be stored.
"""

from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """A data class to represent the artifacts generated during data ingestion."""
    # feature_store_file_path: str
    train_file_path: str
    test_file_path: str