from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class UserBase(BaseModel):
    """Base user model."""
    email: EmailStr
    display_name: Optional[str] = None


class UserCreate(UserBase):
    """User creation model."""
    firebase_uid: str


class User(UserBase):
    """User model with ID."""
    id: str
    firebase_uid: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserInDB(User):
    """User model stored in database."""
    pass


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # Firebase UID
    email: Optional[str] = None
    exp: Optional[int] = None
