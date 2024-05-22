from pydantic import Base64UrlStr, BaseModel


class DetectionBase(BaseModel):
    image: Base64UrlStr
    label: str
    conf: float
    box: list[float]


class DetectionRequest(DetectionBase): ...


class DetectionResponse(DetectionBase):
    id: int

    class Config:
        orm_mode = True
