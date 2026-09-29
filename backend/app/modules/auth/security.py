from datetime import datetime, timedelta,timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import  CryptContext
from app.config import settings


pwd_context =CryptContext(schemes=['bcrypt'], deprecated ='auto')

def hash_password(password :str):
    return pwd_context.hash(password)

def verify_password(plain_password :str, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(subject :str | int,
                        expires_delta: timedelta |None = None,
                        extra_claims : dict[str,Any] | None =None):
    if expires_delta is None:
        expires_delta = timedelta(minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    expire = datetime.now(timezone.utc) + expires_delta

    payload : dict[str, Any] = {
        'sub' : str(subject),
        'exp' : expire,
        'iat' :datetime.now(timezone.utc)
    }

    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(payload, settings.SECRET_KEY, algorithm = settings.ALGORITHM)

def decode_access_token(token : str):
    try :
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms= [settings.ALGORITHM]
        )
        return payload
    except JWTError:
        return None