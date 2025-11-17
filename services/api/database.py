import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from bson import ObjectId
from datetime import datetime
from typing import List, Dict, Optional
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure, OperationFailure

load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "nlp_library")

# Lazy initialization for testing
_client = None
_db = None


def get_client():
    global _client
    if _client is None:
        # Check if using MongoDB Atlas (mongodb+srv://)
        if MONGODB_URI.startswith("mongodb+srv://"):
            # For mongodb+srv://, TLS is automatic
            # Ensure connection string has required parameters
            uri = MONGODB_URI
            if "retryWrites" not in uri:
                separator = "&" if "?" in uri else "?"
                uri = f"{uri}{separator}retryWrites=true&w=majority"
            
            # Use tlsAllowInvalidCertificates to bypass SSL certificate validation
            # This helps with TLS handshake errors in development environments
            # WARNING: Only use this in development, not production!
            # Check environment to ensure this is only used in development
            env = os.getenv("ENV", "development")
            if env == "development":
                _client = AsyncIOMotorClient(
                    uri,
                    tlsAllowInvalidCertificates=True,
                    serverSelectionTimeoutMS=30000,
                    connectTimeoutMS=30000,
                    socketTimeoutMS=30000,
                )
            else:
                # Production: use proper certificate validation
                _client = AsyncIOMotorClient(
                    uri,
                    serverSelectionTimeoutMS=30000,
                    connectTimeoutMS=30000,
                    socketTimeoutMS=30000,
                )
        else:
            # For local MongoDB, no SSL needed
            _client = AsyncIOMotorClient(MONGODB_URI)
    return _client


def get_db():
    global _db
    if _db is None:
        _db = get_client()[DATABASE_NAME]
    return _db


# Collections
def get_documents():
    return get_db().documents


def get_document_stats():
    return get_db().document_stats


def get_users():
    return get_db().users


async def get_user_documents(user_id: str) -> List[Dict]:
    """Get all documents for a user"""
    cursor = get_documents().find({"user_id": user_id}).sort("uploaded_at", -1)
    docs_list = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        # Convert stats_id to string if it exists
        if "stats_id" in doc and doc["stats_id"] is not None:
            doc["stats_id"] = str(doc["stats_id"])
        # Convert datetime to string for JSON serialization
        if "uploaded_at" in doc and doc["uploaded_at"]:
            doc["uploaded_at"] = doc["uploaded_at"].isoformat()
        docs_list.append(doc)
    return docs_list


async def create_document(user_id: str, filename: str) -> str:
    """Create a new document and return its ID"""
    item = {
        "user_id": user_id,
        "filename": filename,
        "uploaded_at": datetime.utcnow(),
        "status": "processing",
        "error": None,
    }
    result = await get_documents().insert_one(item)
    return str(result.inserted_id)


async def update_document(
    document_id: str, stats_id: str = None, status: str = "completed", error: str = None
):
    """Update document with stats reference and status"""
    update_data = {"status": status}
    if stats_id:
        update_data["stats_id"] = ObjectId(stats_id)
    if error:
        update_data["error"] = error

    await get_documents().update_one(
        {"_id": ObjectId(document_id)}, {"$set": update_data}
    )


async def save_document_stats(stats_data: Dict) -> str:
    """Save document stats and return the stats ID"""
    result = await get_document_stats().insert_one(stats_data)
    return str(result.inserted_id)


async def get_document(document_id: str, user_id: str) -> Optional[Dict]:
    """Get a specific document with its stats"""
    item = await get_documents().find_one(
        {"_id": ObjectId(document_id), "user_id": user_id}
    )

    if not item:
        return None

    # Convert ObjectId to string
    item["_id"] = str(item["_id"])
    # Convert stats_id to string if it exists
    if "stats_id" in item and item["stats_id"] is not None:
        item["stats_id"] = str(item["stats_id"])

    # Get stats if available
    if "stats_id" in item:
        stats = await get_document_stats().find_one({"_id": ObjectId(item["stats_id"])})
        if stats:
            stats["_id"] = str(stats["_id"])
            # Recursively convert any ObjectIds in stats to strings
            stats = _convert_objectids_recursive(stats)
            item["stats"] = stats

    return item

async def update_document_metadata(
    document_id: str, user_id: str, filename: str = None
) -> bool:
    """Update document metadata like filename"""
    update_data = {}
    if filename:
        update_data["filename"] = filename
    
    if not update_data:
        return False
    
    result = await get_documents().update_one(
        {"_id": ObjectId(document_id), "user_id": user_id},
        {"$set": update_data}
    )
    return result.modified_count > 0


async def delete_document_by_id(document_id: str, user_id: str) -> bool:
    """Delete a document and its associated stats"""
    document = await get_documents().find_one(
        {"_id": ObjectId(document_id), "user_id": user_id}
    )
    
    if not document:
        return False
    
    if "stats_id" in document and document["stats_id"]:
        await get_document_stats().delete_one({"_id": document["stats_id"]})
    
    result = await get_documents().delete_one(
        {"_id": ObjectId(document_id), "user_id": user_id}
    )
    
    return result.deleted_count > 0



def _convert_objectids_recursive(obj):
    """Recursively convert ObjectIds to strings in nested structures"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: _convert_objectids_recursive(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_objectids_recursive(item) for item in obj]
    else:
        return obj


# User management
async def create_user(email: str, hashed_password: str) -> str:
    """Create a new user and return user ID"""
    try:
        user = {
            "email": email,
            "hashed_password": hashed_password,
            "created_at": datetime.utcnow(),
        }
        result = await get_users().insert_one(user)
        return str(result.inserted_id)
    except (ServerSelectionTimeoutError, ConnectionFailure, OperationFailure) as e:
        # Database connection error - log and re-raise to be handled by the API endpoint
        print(f"Database connection error in create_user: {e}")
        raise  # Re-raise to be handled by the API endpoint
    except Exception as e:
        # Other errors - log and re-raise
        print(f"Unexpected error in create_user: {e}")
        raise  # Re-raise to be handled by the API endpoint


async def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email"""
    try:
        user = await get_users().find_one({"email": email})
        if user:
            user["_id"] = str(user["_id"])
            # Convert datetime to string for JSON serialization
            if "created_at" in user and user["created_at"]:
                user["created_at"] = user["created_at"].isoformat()
        return user
    except (ServerSelectionTimeoutError, ConnectionFailure, OperationFailure) as e:
        # Database connection error - log and re-raise to be handled by the API endpoint
        print(f"Database connection error in get_user_by_email: {e}")
        raise  # Re-raise to be handled by the API endpoint
    except Exception as e:
        # Other errors - log and re-raise
        print(f"Unexpected error in get_user_by_email: {e}")
        raise  # Re-raise to be handled by the API endpoint


async def get_user_by_id(user_id: str) -> Optional[Dict]:
    """Get user by ID"""
    try:
        user = await get_users().find_one({"_id": ObjectId(user_id)})
        if user:
            user["_id"] = str(user["_id"])
            # Convert datetime to string
            if "created_at" in user and user["created_at"]:
                user["created_at"] = user["created_at"].isoformat()
        return user
    except (ServerSelectionTimeoutError, ConnectionFailure, OperationFailure) as e:
        # Database connection error - log and return None
        print(f"Database connection error in get_user_by_id: {e}")
        raise  # Re-raise to be handled by the API endpoint
    except Exception as e:
        # Other errors - log and return None
        print(f"Unexpected error in get_user_by_id: {e}")
        raise  # Re-raise to be handled by the API endpoint


async def test_connection(timeout: float = 5.0):
    """Test MongoDB connection with timeout"""
    import asyncio
    try:
        # Use asyncio.wait_for to add a timeout
        await asyncio.wait_for(
            get_client().admin.command("ping"),
            timeout=timeout
        )
        print("Connected to MongoDB successfully")
        return True
    except asyncio.TimeoutError:
        print(f"MongoDB connection test timed out after {timeout} seconds")
        print("The server will continue, but database operations may fail.")
        return False
    except Exception as e:
        error_msg = str(e)
        print(f"MongoDB connection failed: {e}")
        
        # Provide helpful error messages for common issues
        if "TLSV1_ALERT_INTERNAL_ERROR" in error_msg or "SSL handshake failed" in error_msg:
            print("\n⚠️  SSL/TLS Handshake Error Detected")
            print("This is often caused by Python/OpenSSL compatibility issues.")
            print("\nPossible solutions:")
            print("1. Update your Python and OpenSSL:")
            print("   - macOS: brew upgrade python@3.11 openssl")
            print("   - Linux: sudo apt-get update && sudo apt-get upgrade python3.11 openssl")
            print("\n2. Try using a standard connection string instead of mongodb+srv://")
            print("   Get it from MongoDB Atlas: Connect > Drivers > Python")
            print("   Use the 'Standard connection string' option")
            print("\n3. Check your network/firewall settings")
            print("   Ensure port 27017 (or 443 for SRV) is not blocked")
            print("\n4. Verify your MongoDB Atlas network access settings")
            print("   Ensure your IP is whitelisted (or use 0.0.0.0/0 for development)")
        
        return False
