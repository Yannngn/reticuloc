from pydantic import Base64UrlStr, BaseModel


class ImageBase(BaseModel):
    image: Base64UrlStr


class ImageRequest(ImageBase): ...


class ImageResponse(ImageBase):
    id: int

    class Config:
        orm_mode = True
