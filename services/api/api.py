# FastAPI backend API
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
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


def convert_to_native_types(obj):
    """
    Recursively convert numpy types, MongoDB ObjectIds, datetime objects, and other non-JSON-serializable types to native Python types.
    """
    from bson import ObjectId
    
    # Handle MongoDB ObjectId first (must be before dict/list checks)
    if isinstance(obj, ObjectId):
        return str(obj)
    
    # Handle datetime objects
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # Handle numpy types
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    
    # Recursively handle collections
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Test database connection on startup (non-blocking with timeout)"""
    # Test connection with timeout - don't block startup if it fails
    import asyncio
    try:
        # Run connection test with 5 second timeout - won't block startup
        await asyncio.wait_for(test_connection(), timeout=5.0)
    except asyncio.TimeoutError:
        print("Warning: MongoDB connection test timed out. Server will start anyway.")
        print("Database operations may fail until connection is established.")
    except Exception as e:
        # Log but don't fail startup
        print(f"Warning: Could not test MongoDB connection on startup: {e}")
        print("Server will start, but database operations may fail until connection is established.")
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
        # Convert any remaining ObjectIds to strings recursively
        documents = convert_to_native_types(documents)
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

        # Process the document (skip CSV generation for API - we store everything in MongoDB)
        analysis_results = process_path(ipath=Path(temp_path), outdir=temp_outdir, from_raw=True, write_csv=False)

        # Convert numpy types and ensure analysis results are JSON serializable before saving to database
        import json
        serialized_stats = convert_to_native_types(analysis_results)
        
        # Verify JSON serialization
        try:
            json.dumps(serialized_stats)
        except (TypeError, ValueError):
            # If serialization still fails, convert problematic objects to strings
            fallback_stats = {}
            for key, value in serialized_stats.items():
                try:
                    json.dumps(value)
                    fallback_stats[key] = value
                except (TypeError, ValueError):
                    fallback_stats[key] = str(value)
            serialized_stats = fallback_stats

        # Save stats to database
        stats_id = await save_document_stats(serialized_stats)

        # Update document with stats
        await update_document(document_id, stats_id, "completed")

        # Clean up temp file
        temp_path.unlink()

        # Ensure the entire response is JSON serializable (convert any remaining ObjectIds)
        response_data = {
            "document_id": document_id,
            "filename": file.filename,
            "processing_status": "completed",
            "stats": serialized_stats,
        }
        # Double-check everything is serializable
        response_data = convert_to_native_types(response_data)

        return response_data

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

        # Convert any remaining ObjectIds, datetime objects, and non-serializable types to native types recursively
        document = convert_to_native_types(document)

        return document
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")
    
@app.put("/documents/{user_id}/{document_id}")
async def update_document_endpoint(
    user_id: str,
    document_id: str,
    filename: str = None,
    current_user_id: str = Depends(get_current_user)
):
    """Update document metadata (e.g., rename)"""
    # Verify the user owns this document
    if current_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to update this document"
        )
    
    try:
        from services.api.database import update_document_metadata
        
        success = await update_document_metadata(
            document_id, user_id, filename=filename
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Document not found or no changes made"
            )
        
        updated_doc = await get_document(document_id, user_id)
        if not updated_doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        updated_doc = convert_to_native_types(updated_doc)
        return updated_doc
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update document: {str(e)}"
        )


@app.delete("/documents/{user_id}/{document_id}")
async def delete_document_endpoint(
    user_id: str,
    document_id: str,
    current_user_id: str = Depends(get_current_user)
):
    """Delete a document and its associated stats"""
    if current_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to delete this document"
        )
    
    try:
        from services.api.database import delete_document_by_id
        
        success = await delete_document_by_id(document_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "success": True,
            "message": "Document deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )


# Authentication endpoints
@app.post("/auth/register", response_model=User)
async def register(user: UserCreate):
    """Register a new user"""
    from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure, OperationFailure
    
    try:
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
    except (ServerSelectionTimeoutError, ConnectionFailure, OperationFailure) as e:
        # Database connection error - return 503 Service Unavailable
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable. Please try again later."
        )
    except HTTPException:
        # Re-raise HTTP exceptions (like 400)
        raise
    except Exception as e:
        # Other unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during registration: {str(e)}"
        )


@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Login user and return access token"""
    from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure, OperationFailure
    
    try:
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
    except (ServerSelectionTimeoutError, ConnectionFailure, OperationFailure) as e:
        # Database connection error - return 503 Service Unavailable
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable. Please try again later."
        )
    except HTTPException:
        # Re-raise HTTP exceptions (like 401)
        raise
    except Exception as e:
        # Other unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during login: {str(e)}"
        )


@app.get("/auth/me", response_model=User)
async def get_current_user_info(current_user_id: str = Depends(get_current_user)):
    """Get current user information"""
    from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure, OperationFailure
    
    try:
        user = await get_user_by_id(current_user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Parse the ISO string back to datetime if necessary
        created_at = user["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return User(
            id=user["_id"],
            email=user["email"],
            created_at=created_at,
        )
    except (ServerSelectionTimeoutError, ConnectionFailure, OperationFailure) as e:
        # Database connection error - return 503 Service Unavailable
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable. Please try again later."
        )
    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    except Exception as e:
        # Other unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while retrieving user information: {str(e)}"
        )
