from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_weights: str
    params_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    training_data: Path
    validation_data: Path
    all_image_path: list
    params_epochs: int
    params_weights: str
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_learning_rate: float
    params_classes: int
    params_device: str


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    testing_data: Path
    all_params: dict
    all_image_path: list
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
    params_device: str
    params_learning_rate: float
    params_classes: int
    params_weights: str
