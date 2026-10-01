"""
Shared FastAPI dependencies.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.security import decode_access_token
from app.database import get_db
from app.modules.auth.model import User
from app.modules.auth.service import get_user_by_id


# ============ BEARER SCHEME ============
security = HTTPBearer(auto_error=True)   # ← auto_error=True


# ============ GET CURRENT USER ============
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    JWT token verify karke current user return karo.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Defensive check (auto_error=True ke saath bhi safety)
    if credentials is None:
        raise credentials_exception

    token = credentials.credentials

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise credentials_exception

    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        raise credentials_exception

    user = await get_user_by_id(db, user_id)
    if user is None:
        raise credentials_exception

    return user


# ============ GET CURRENT ACTIVE USER ============
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