from pydantic import BaseModel

class UserCreate(BaseModel):
    full_name: str
    phone_number: str
    password: str

class UserLogin(BaseModel):
    phone_number: str
    password: str

class UserOut(BaseModel):
    id: int
    full_name: str
    phone_number: str
    model_config = {"from_attributes": True}
