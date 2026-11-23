from dataclasses import dataclass


@dataclass
class AbstractionConfig:
    """New incident edges per character and shared edges per character pair."""

    incremental_enabled: bool = True
    interval_node: int = 50
    interval_pair: int = 30
    final_lower_bound_node: int = 10
    final_lower_bound_pair: int = 3
