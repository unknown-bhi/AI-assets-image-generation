from datetime import datetime
from pydantic import BaseModel,ConfigDict,EmailStr,Field

class UserRegister(BaseModel):
    username :str =Field(...,min_length =3, max_length=50)
    email : EmailStr
    password :str =Field(...,min_length=6, max_length=128)

class UserLogin(BaseModel):
    username: str
    password : str

class UserOut(BaseModel):

    model_config =ConfigDict(from_attributes=True)

    id: int
    username: str
    email: EmailStr
    is_active: bool
    is_superuser: bool
    daily_quota: int
    created_at: datetime

class Token(BaseModel):
    access_token : str
    token_type :str = 'bearer'

class TokenPayload(BaseModel):
    sub :str
    exp :int