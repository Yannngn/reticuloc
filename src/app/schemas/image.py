from pydantic import BaseModel


class ImageBase(BaseModel):
    data: str
