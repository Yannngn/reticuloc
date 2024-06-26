from pydantic import BaseModel


class DetectionBase(BaseModel):
    box: list[float]
    label: str
    conf: float
