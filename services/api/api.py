# fastapi backend API
from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from services.api import routes
from pathlib import Path
import tempfile
import os

try:
    from services.nlp.analyze_texts import process_path
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from services.nlp.analyze_texts import process_path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router)

# post endpoint for user to upload new document for nlp processing

@app.post("/process_document/")
async def process_document(file: UploadFile = File(...)):
    # save as a temporary file
    temp_dir = tempfile.TemporaryDirectory()
    temp = os.path.join(temp_dir, file.filename)
    try:
        contents = file.file.read()
        temp.write(contents)
        temp.flush()

        temp_path = Path(temp.name)
        temp_dir = temp_path.parent
        temp_outdir = temp_dir / "nlp_outputs"

        process_path(ipath=temp_path, outdir=temp_outdir, from_raw=True)
        # TODO: return actual results from database
        return {"status": "success", "output_dir": str(temp_outdir)}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read uploaded file")
    finally:
        file.file.close()
    