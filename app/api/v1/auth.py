from fastapi import APIRouter, HTTPException, Depends, Header, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import firebase_admin
from firebase_admin import auth, credentials
from app.core.config import settings
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Add imports for API Key support
from app.db.mongo import get_database
import secrets
from datetime import datetime
from fastapi import Body

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Initialize Firebase Admin SDK
def init_firebase():
    """Initialize Firebase Admin SDK if not already initialized."""
    if firebase_admin._apps:
        return True
    
    firebase_creds_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
    
    # 1. Try loading from JSON string in environment variable (Best for Cloud)
    if firebase_creds_json:
        try:
            # Check if it's already a dict or needs parsing
            import json
            if isinstance(firebase_creds_json, str):
                creds_dict = json.loads(firebase_creds_json)
            else:
                creds_dict = firebase_creds_json
                
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialized from FIREBASE_CREDENTIALS_JSON")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Firebase from JSON env var: {e}")

    # 2. Fallback to file path
    firebase_creds_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "rag-x-firebase-credentials.json")
    
    # Try different possible paths
    possible_paths = [
        firebase_creds_path,
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), firebase_creds_path),
        os.path.join(os.getcwd(), firebase_creds_path),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                cred = credentials.Certificate(path)
                firebase_admin.initialize_app(cred)
                logger.info(f"Firebase Admin SDK initialized with: {path}")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize Firebase with {path}: {e}")
    
    logger.warning(f"Firebase credentials not found. Tried env var and paths: {possible_paths}")
    return False

# Initialize Firebase on module load
firebase_initialized = init_firebase()

security = HTTPBearer(auto_error=False)

# For testing without Firebase - set TEST_MODE=true in .env
TEST_USER_ID = "test_user_123"


async def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_test_user: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> str:
    """
    Verify Firebase ID token and return user ID.
    
    This dependency extracts the user ID from the Firebase auth token,
    which is used to namespace embeddings in Pinecone.
    
    For testing: Set TEST_MODE=true in .env and use X-Test-User header.
    """
    # Test mode bypass for Postman testing
    if settings.TEST_MODE:
        if x_test_user:
            return x_test_user
        return TEST_USER_ID
    


    
    # Check if Firebase is initialized
    if not firebase_initialized:
        logger.error("Firebase Admin SDK not initialized")
        raise HTTPException(
            status_code=500,
            detail="Authentication service not configured"
        )
    
    
    # 3. Check for API Key
    if x_api_key:
        try:
            db = await get_database()
            key_doc = await db.api_keys.find_one({"key": x_api_key})
            if key_doc:
                return key_doc["user_id"]
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            pass # Fall through to Bearer check

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    
    token = credentials.credentials
    
    try:
        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token["uid"]
        logger.info(f"Authenticated user: {user_id}")
        return user_id
    
    except auth.InvalidIdTokenError as e:
        logger.error(f"Invalid token: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    except auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired"
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials"
        )


async def get_optional_user_id(
    authorization: Optional[str] = Header(None)
) -> Optional[str]:
    """
    Optionally verify Firebase ID token.
    Returns None if no token is provided.
    """
    if not authorization:
        return None
    
    try:
        # Extract token from "Bearer <token>"
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            return None
        
        decoded_token = auth.verify_id_token(token)
        return decoded_token["uid"]
    
    except Exception:
        return None


@router.get("/verify")
async def verify_token(user_id: str = Depends(get_current_user_id)):
    """Verify the current user's token."""
    return {
        "valid": True,
        "user_id": user_id
    }


@router.get("/me")
async def get_current_user(user_id: str = Depends(get_current_user_id)):
    """Get current user information."""
    try:
        user = auth.get_user(user_id)
        return {
            "uid": user.uid,
            "email": user.email,
            "display_name": user.display_name,
            "photo_url": user.photo_url,
            "email_verified": user.email_verified
        }
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(status_code=404, detail="User not found")


# API Key Management
@router.post("/api-key")
async def generate_api_key(user_id: str = Depends(get_current_user_id)):
    """Generate or regenerate an API Key for the user."""
    db = await get_database()
    
    # Generate a secure key
    new_key = f"ragx_{secrets.token_urlsafe(32)}"
    
    # Store in DB (upsert)
    await db.api_keys.update_one(
        {"user_id": user_id},
        {"$set": {
            "key": new_key, 
            "user_id": user_id, 
            "updated_at": datetime.utcnow(),
            "created_at": datetime.utcnow()
        }},
        upsert=True
    )
    
    return {"api_key": new_key}

@router.get("/api-key")
async def get_api_key_endpoint(user_id: str = Depends(get_current_user_id)):
    """Get the current API Key for the user."""
    db = await get_database()
    
    key_doc = await db.api_keys.find_one({"user_id": user_id})
    
    if not key_doc:
        # Auto-generate if not exists
        return await generate_api_key(user_id=user_id)
        
    return {"api_key": key_doc["key"]}

