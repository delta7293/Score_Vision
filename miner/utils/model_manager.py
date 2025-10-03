from pathlib import Path
from typing import Dict, Optional
from ultralytics import YOLO
from loguru import logger

from miner.utils.device import get_optimal_device
from scripts.download_models import download_models


class ModelManager:
    """Manages the loading and caching of YOLO models with multi-GPU support."""

    def __init__(self, device: Optional[str] = None):
        # 기본 디바이스 설정 (fallback 용도)
        self.default_device = get_optimal_device(device)
        # 캐시 구조를 model_name + device 조합으로 관리
        self.device = self.default_device
        self.models: Dict[str, Dict[str, YOLO]] = {}
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Define model paths
        self.model_paths = {
            "player": self.data_dir / "football-player-detection.pt",
            "pitch": self.data_dir / "football-pitch-detection.pt",
            "ball": self.data_dir / "football-ball-detection.pt"
        }

        # Check if models exist, download if missing
        self._ensure_models_exist()

    def _ensure_models_exist(self) -> None:
        """Check if required models exist, download if missing."""
        missing_models = [
            name for name, path in self.model_paths.items()
            if not path.exists()
        ]

        if missing_models:
            logger.info(f"Missing models: {', '.join(missing_models)}")
            logger.info("Downloading required models...")
            download_models()

    def load_model(self, model_name: str, device: Optional[str] = None) -> YOLO:
        """
        Load a model by name onto a specific device, using cache if available.

        Args:
            model_name: Name of the model to load ('player', 'pitch', or 'ball')
            device: Optional device (e.g. 'cuda:0', 'cuda:1', 'cpu', 'mps')

        Returns:
            YOLO: The loaded model
        """
        if model_name not in self.model_paths:
            raise ValueError(f"Unknown model: {model_name}")

        target_device = get_optimal_device(device or self.default_device)

        # 캐시 확인 (모델+디바이스 조합)
        if model_name in self.models and target_device in self.models[model_name]:
            return self.models[model_name][target_device]

        model_path = self.model_paths[model_name]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Please ensure all required models are downloaded."
            )

        logger.info(f"Loading {model_name} model from {model_path} to {target_device}")
        model = YOLO(str(model_path)).to(device=target_device)

        if model_name not in self.models:
            self.models[model_name] = {}
        self.models[model_name][target_device] = model
        return model

    def load_all_models(self, device: Optional[str] = None) -> None:
        """Load all models into cache on a given device."""
        for model_name in self.model_paths.keys():
            self.load_model(model_name, device=device)

    def get_model(self, model_name: str, device: Optional[str] = None) -> YOLO:
        """
        Get a model by name, loading it if necessary.

        Args:
            model_name: Name of the model to get ('player', 'pitch', or 'ball')
            device: Optional device (e.g. 'cuda:0', 'cuda:1', 'cpu', 'mps')

        Returns:
            YOLO: The requested model
        """
        return self.load_model(model_name, device=device)

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self.models.clear()
