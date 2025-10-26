# FastAPI backend API
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import tempfile
from datetime import timedelta, datetime

# Import NLP processing
try:
    from services.nlp.analyze_texts import process_path
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from services.nlp.analyze_texts import process_path

# Import database functions
from services.api.database import (
    get_user_documents,
    create_document,
    update_document,
    save_document_stats,
    get_document,
    test_connection,
    create_user,
    get_user_by_email,
    get_user_by_id,
)

# Import authentication
from services.api.auth import (
    UserCreate,
    UserLogin,
    User,
    Token,
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Test database connection on startup"""
    await test_connection()
    yield


app = FastAPI(title="NLP Document Library API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "NLP Document Library API - Test Version", "version": "1.0.0"}


@app.get("/documents/{user_id}")
async def get_documents(user_id: str):
    """Get all documents for a user"""
    try:
        documents = await get_user_documents(user_id)
        return {"user_id": user_id, "documents": documents}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get documents: {str(e)}"
        )


@app.options("/documents/upload")
async def upload_document_options():
    """Handle CORS preflight for upload endpoint"""
    return {"message": "OK"}


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...), user_id: str = Depends(get_current_user)
):
    """Upload and process a document"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Create document first
    document_id = await create_document(user_id, file.filename)

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        # Create output directory for NLP processing
        temp_outdir = temp_path.parent / "nlp_outputs"
        temp_outdir.mkdir(exist_ok=True)

        # Process the document
        meta = process_path(ipath=Path(temp_path), outdir=temp_outdir, from_raw=True)

        # Save stats to database
        stats_id = await save_document_stats(meta)

        # Update document with stats
        await update_document(document_id, stats_id, "completed")

        # Clean up temp file
        temp_path.unlink()

        # Ensure meta is JSON serializable
        try:
            # Try to serialize meta to ensure it's valid JSON
            import json

            json.dumps(meta)
            serialized_meta = meta
        except (TypeError, ValueError) as e:
            # If serialization fails, convert problematic objects to strings
            serialized_meta = {}
            for key, value in meta.items():
                try:
                    json.dumps(value)
                    serialized_meta[key] = value
                except (TypeError, ValueError):
                    serialized_meta[key] = str(value)

        return {
            "document_id": document_id,
            "filename": file.filename,
            "processing_status": "completed",
            "stats": serialized_meta,
        }

    except Exception as e:
        # Update document with error status
        await update_document(document_id, status="failed", error=str(e))

        # Clean up temp file if it exists
        if "temp_path" in locals():
            try:
                temp_path.unlink()
            except Exception:
                pass

        raise HTTPException(
            status_code=500, detail=f"Document processing failed: {str(e)}"
        )


@app.get("/documents/{user_id}/{document_id}")
async def get_document_by_id(user_id: str, document_id: str):
    """Get a specific document with its stats"""
    try:
        document = await get_document(document_id, user_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Convert datetime to string for JSON serialization
        if "uploaded_at" in document and document["uploaded_at"]:
            document["uploaded_at"] = document["uploaded_at"].isoformat()

        return document
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


# Authentication endpoints
@app.post("/auth/register", response_model=User)
async def register(user: UserCreate):
    """Register a new user"""
    existing_user = await get_user_by_email(user.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    hashed_password = get_password_hash(user.password)
    user_id = await create_user(user.email, hashed_password)

    created_user = await get_user_by_id(user_id)
    # Return user info (without password)
    return User(
        id=user_id,
        email=user.email,
        created_at=created_user["created_at"],
    )


@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Login user and return access token"""
    # Get user from database
    user = await get_user_by_email(user_credentials.email)
    if not user or not verify_password(
        user_credentials.password, user["hashed_password"]
    ):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token with different expiration based on remember_me
    if user_credentials.remember_me:
        # Extended token for 30 days when remember me is checked
        access_token_expires = timedelta(days=30)
    else:
        # Standard token expiration
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    access_token = create_access_token(
        data={"sub": user["_id"]}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/auth/me", response_model=User)
async def get_current_user_info(current_user_id: str = Depends(get_current_user)):
    """Get current user information"""
    user = await get_user_by_id(current_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Parse the ISO string back to datetime
    created_at = datetime.fromisoformat(user["created_at"])

    return User(
        id=user["_id"],
        email=user["email"],
        created_at=created_at,
    )
