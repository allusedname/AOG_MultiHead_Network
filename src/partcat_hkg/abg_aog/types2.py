from dataclasses import dataclass

@dataclass
class V7Query:
    sample_index: int
    part_id: int
    part_name: str
    priority: float
