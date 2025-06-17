# config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class PipelineConfig:
    timegpt_token: str
    top_n_products: int = 10
    test_days: int = 14
    forecast_horizon: int = 7
    confidence_levels: List[int] = field(default_factory=lambda: [80, 90])

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        import json
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
