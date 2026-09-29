from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.auth.security import create_access_token,hash_password,verify_password
from app.modules.auth.schema import Token, UserLogin,UserRegister
from app.modules.auth.model import User
from app.modules.auth.schema import Token ,UserRegister,UserLogin

async def get_user_by_username(db: AsyncSession, username:str):
    result =await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()

async def get_user_by_email( db: AsyncSession, email: str) :
   
    result = await db.execute(
        select(User).where(User.email == email)
    )
    return result.scalar_one_or_none()



async def get_user_by_id(db: AsyncSession, user_id: int):
   
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, data: UserRegister):

  
    existing = await get_user_by_username(db, data.username)
    if existing:
        raise ValueError("Username already registered")


    existing = await get_user_by_email(db, data.email)
    if existing:
        raise ValueError("Email already registered")


    user = User(
        username=data.username,
        email=data.email,
        hashed_password=hash_password(data.password),
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)   

    return user


async def authenticate_user( db: AsyncSession, data: UserLogin) :
    user = await get_user_by_username(db, data.username)
    if not user:
        return None

    if not verify_password(data.password, user.hashed_password):
        return None

    if not user.is_active:
        return None

    return user


def create_token_for_user(user: User) :
  
    token = create_access_token(
        subject=user.id,
        extra_claims={"username": user.username},
    )
    return Token(access_token=token, token_type="bearer")

