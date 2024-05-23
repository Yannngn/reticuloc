from pydantic import BaseModel


class DetectionCropsBase(BaseModel):
    data: str
    box: list[float]
    label: str
    conf: float
