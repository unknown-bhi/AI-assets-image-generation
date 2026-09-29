
from fastapi import Depends, HTTPException, status,Security
from fastapi.security import HTTPAuthorizationCredentials,HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.security import decode_access_token
from app.database import get_db
from app.modules.auth.model import User
from app.modules.auth.service import get_user_by_id



security =HTTPBearer(auto_error=False)


async def get_current_user(credentials :HTTPAuthorizationCredentials | None = Security(security),
                            token: str = Depends(security), db: AsyncSession = Depends(get_db)) :

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # 1. Token decode karo
    token =credentials.credentials
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    # 2. User ID nikalo
    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise credentials_exception

    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        raise credentials_exception

    # 3. DB se user fetch karo
    user = await get_user_by_id(db, user_id)
    if user is None:
        raise credentials_exception

    return user



async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Sirf active users ko allow karo."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user