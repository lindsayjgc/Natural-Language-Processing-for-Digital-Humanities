# FastAPI backend API
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile

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
    create_library_item,
    update_library_item,
    save_document_stats,
    get_library_item,
    test_connection,
)

app = FastAPI(title="NLP Document Library API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Test database connection on startup"""
    await test_connection()


@app.get("/")
async def root():
    return {"message": "NLP Document Library API", "version": "1.0.0"}


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


@app.post("/documents/upload")
async def upload_document(user_id: str = Form(...), file: UploadFile = File(...)):
    """Upload and process a document"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Create library item first
    item_id = await create_library_item(user_id, file.filename)

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
        meta = process_path(ipath=temp_path, outdir=temp_outdir, from_raw=True)

        # Save stats to database
        stats_id = await save_document_stats(meta)

        # Update library item with stats
        await update_library_item(item_id, stats_id, "completed")

        # Clean up temp file
        temp_path.unlink()

        return {
            "library_item_id": item_id,
            "filename": file.filename,
            "processing_status": "completed",
            "stats": meta,
        }

    except Exception as e:
        # Update library item with error status
        await update_library_item(item_id, status="failed", error=str(e))

        # Clean up temp file if it exists
        if "temp_path" in locals():
            try:
                temp_path.unlink()
            except Exception:
                pass

        raise HTTPException(
            status_code=500, detail=f"Document processing failed: {str(e)}"
        )


@app.get("/documents/{user_id}/{item_id}")
async def get_document(user_id: str, item_id: str):
    """Get a specific document with its stats"""
    try:
        document = await get_library_item(item_id, user_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return document
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")
