from pydantic import Base64UrlStr, BaseModel


class UserBase(BaseModel):
    image: Base64UrlStr


class UserRequest(UserBase): ...


class UserResponse(UserBase):
    id: int

    class Config:
        orm_mode = True
