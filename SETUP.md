# Detailed Setup Guide

Complete installation and configuration instructions for the NLP Document Library.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Python Setup](#python-setup)
3. [MongoDB Atlas Configuration](#mongodb-atlas-configuration)
4. [Backend Setup](#backend-setup)
5. [Frontend Setup](#frontend-setup)
6. [Troubleshooting](#troubleshooting)

## System Requirements

### Operating Systems
- macOS 10.15+
- Linux (Ubuntu 20.04+, similar distributions)
- Windows 10/11 (WSL2 recommended for best compatibility)

### Software Prerequisites
- **Python 3.11** or higher (required for scientific libraries)
- **Node.js 18** or higher
- **pnpm** package manager (or npm/yarn)
- **Git** for version control
- **MongoDB Atlas account** (free tier)

### Hardware Recommendations
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space for dependencies
- **CPU**: Multi-core processor recommended for NLP processing

## Python Setup

### 1. Install Python 3.11

#### macOS
```bash
# Using Homebrew
brew install python@3.11

# Verify installation
python3.11 --version
```

#### Linux (Ubuntu/Debian)
```bash
# Add deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev

# Verify installation
python3.11 --version
```

#### Windows
1. Download Python 3.11 from [python.org](https://www.python.org/downloads/)
2. Run installer with "Add Python to PATH" checked
3. Verify in Command Prompt: `python --version`

### 2. Create Virtual Environment

```bash
# Navigate to project directory
cd Natural-Language-Processing-for-Digital-Humanities

# Create virtual environment
python3.11 -m venv venv311

# Activate virtual environment
# macOS/Linux:
source venv311/bin/activate

# Windows:
venv311\Scripts\activate

# Your prompt should now show (venv311)
```

### 3. Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# This will install:
# - pandas, numpy (data processing)
# - nltk, spacy, transformers, torch (NLP)
# - fastapi, uvicorn, motor, pymongo (API/database)
# - docx2txt, python-docx (document processing)
```

### 4. Download NLTK Data

NLTK requires additional data packages for tokenization, tagging, and sentiment analysis:

```bash
python -c "import nltk; \
nltk.download('punkt_tab'); \
nltk.download('stopwords'); \
nltk.download('wordnet'); \
nltk.download('vader_lexicon'); \
nltk.download('averaged_perceptron_tagger_eng'); \
nltk.download('maxent_ne_chunker'); \
nltk.download('words')"
```

### 5. Download spaCy Model

```bash
# Download English language model
python -m spacy download en_core_web_sm
```

## MongoDB Atlas Configuration

### 1. Create MongoDB Atlas Account

1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
2. Sign up for a free account
3. Verify your email address

### 2. Create a Cluster

1. Click "Build a Database"
2. Choose "FREE" tier (M0 Sandbox)
3. Select your cloud provider and region
4. Click "Create Cluster" (takes 3-5 minutes)

### 3. Create Database User

1. Go to "Database Access" in the left sidebar
2. Click "Add New Database User"
3. Choose "Password" authentication
4. Create username and password (save these!)
5. Set user privileges to "Read and write to any database"
6. Click "Add User"

### 4. Configure Network Access

1. Go to "Network Access" in the left sidebar
2. Click "Add IP Address"
3. For development: Click "Allow Access from Anywhere" (0.0.0.0/0)
   - For production: Add specific IP addresses
4. Click "Confirm"

### 5. Get Connection String

1. Go to "Database" in the left sidebar
2. Click "Connect" on your cluster
3. Choose "Connect your application"
4. Copy the connection string
5. Replace `<password>` with your database user password
6. Replace `<dbname>` with `nlp_library` (or your preferred name)

Example connection string:
```
mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
```

## Backend Setup

### 1. Configure Environment Variables

```bash
# Copy environment template
cp env.example .env

# Edit .env file
nano .env  # or use your preferred editor
```

Add your MongoDB Atlas connection string:
```env
# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
DATABASE_NAME=nlp_library
```

### 2. Test API Connection

```bash
# Navigate to API directory
cd services/api

# Test MongoDB connection
../../venv311/bin/python -c "
from database import test_connection
import asyncio
asyncio.run(test_connection())
"

# Expected output: "Connected to MongoDB successfully"
```

### 3. Start API Server

```bash
# From services/api directory
../../venv311/bin/uvicorn api:app --reload --port 8000

# Server will start at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

### 4. Test API Endpoints

Open a new terminal and test:

```bash
# Health check
curl http://localhost:8000/

# List documents (should return empty array initially)
curl http://localhost:8000/documents/demo_user

# Upload a test document
curl -X POST -F "user_id=demo_user" -F "file=@test.txt" \
  http://localhost:8000/documents/upload
```

## Frontend Setup

### 1. Install Node.js and pnpm

#### macOS
```bash
# Install Node.js with Homebrew
brew install node

# Install pnpm globally
npm install -g pnpm

# Verify installations
node --version
pnpm --version
```

#### Linux
```bash
# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install pnpm
npm install -g pnpm
```

#### Windows
1. Download Node.js from [nodejs.org](https://nodejs.org/)
2. Run installer
3. Install pnpm: `npm install -g pnpm`

### 2. Install Frontend Dependencies

```bash
# Navigate to frontend directory
cd services/frontend

# Install dependencies
pnpm install

# This installs Next.js, React, TypeScript, Tailwind, and UI components
```

### 3. Configure Environment

```bash
# Create environment file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

### 4. Start Development Server

```bash
# From services/frontend directory
pnpm dev

# Frontend will start at http://localhost:3000
```

### 5. Test Frontend

1. Open http://localhost:3000 in your browser
2. You should see the documents page
3. Try uploading a text file
4. Verify it appears in the document list
5. Click "View Analysis" to see NLP stats

## Running Both Services

### Quick Start (Recommended)

**Start both services with one command:**
```bash
./start-dev.sh
```

This script will:
- Check prerequisites (virtual environment, Node.js, dependencies)
- Start backend API server on port 8000
- Start frontend development server on port 3000
- Display colored, prefixed logs from both services
- Handle graceful shutdown with Ctrl+C

**Stop both services:**
```bash
./stop-dev.sh
```

This script will:
- Kill processes on ports 8000 and 3000
- Clean up stuck Node.js processes
- Remove log files (use `--keep` flag to preserve them)

**Stop script options:**
```bash
./stop-dev.sh --help     # Show usage information
./stop-dev.sh --keep     # Keep log files after stopping
./stop-dev.sh --verbose  # Show detailed output
```

### Manual Terminal Setup (Alternative)

If you prefer to run services manually, you need two terminal windows/tabs:

**Terminal 1 - Backend**
```bash
cd services/api
../../venv311/bin/uvicorn api:app --reload --port 8000
```

**Terminal 2 - Frontend**
```bash
cd services/frontend
pnpm dev
```

### Access Points
- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Troubleshooting

### Python Issues

**Problem**: `python3.11` not found
```bash
# macOS: Ensure Homebrew Python is in PATH
echo 'export PATH="/usr/local/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Linux: Install from deadsnakes PPA (see above)

# Windows: Reinstall with "Add to PATH" option
```

**Problem**: `pip install` fails with compilation errors
```bash
# Install build tools
# macOS:
xcode-select --install

# Linux:
sudo apt-get install build-essential python3.11-dev

# Then retry pip install
```

### NLTK Issues

**Problem**: NLTK data not found
```bash
# Download all data to user directory
python -c "import nltk; nltk.download('all', download_dir='~/nltk_data')"

# Or download to project venv
python -c "import nltk; nltk.download('all')"
```

**Problem**: SSL certificate error during NLTK download
```bash
# Use alternative download method
python -c "import nltk; nltk.download('punkt_tab', download_dir='~/nltk_data', quiet=False, force=True)"
```

### MongoDB Issues

**Problem**: Connection timeout
- Check network access settings in MongoDB Atlas
- Verify IP address is whitelisted (or use 0.0.0.0/0 for dev)
- Check firewall settings

**Problem**: Authentication failed
- Verify username and password in connection string
- Ensure database user has correct permissions
- Check password for special characters (URL encode if needed)

**Problem**: `pymongo.errors.ServerSelectionTimeoutError`
```bash
# Test connection manually
python -c "
from pymongo import MongoClient
client = MongoClient('YOUR_CONNECTION_STRING')
print(client.server_info())
"
```

### Frontend Issues

**Problem**: `pnpm` command not found
```bash
# Install pnpm globally
npm install -g pnpm

# Or use npx
npx pnpm install
npx pnpm dev
```

**Problem**: Port 3000 already in use
```bash
# Use different port
pnpm dev -- --port 3001

# Or kill process using port 3000
# macOS/Linux:
lsof -ti:3000 | xargs kill -9

# Windows:
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

**Problem**: API connection error (CORS)
- Verify API is running on port 8000
- Check `.env.local` has correct `NEXT_PUBLIC_API_URL`
- Ensure API CORS middleware is configured (already done)

### Script Issues

**Problem**: `./start-dev.sh: Permission denied`
```bash
# Make script executable
chmod +x start-dev.sh
chmod +x stop-dev.sh
```

**Problem**: Script fails with "command not found" errors
```bash
# Ensure you're in the project root directory
pwd  # Should show path ending with "Natural-Language-Processing-for-Digital-Humanities"

# Check if files exist
ls -la start-dev.sh stop-dev.sh
```

**Problem**: Services don't start or ports are busy
```bash
# Stop any existing services
./stop-dev.sh --verbose

# Check what's using the ports
lsof -i :8000
lsof -i :3000

# Kill processes manually if needed
sudo kill -9 $(lsof -ti:8000)  # Backend
sudo kill -9 $(lsof -ti:3000)  # Frontend
```

**Problem**: Script hangs or doesn't show logs
```bash
# Check log files directly
tail -f backend.log
tail -f frontend.log

# Run with verbose output
./start-dev.sh
# Then in another terminal:
tail -f backend.log | sed 's/^/[BACKEND] /'
tail -f frontend.log | sed 's/^/[FRONTEND] /'
```

### General Issues

**Problem**: Virtual environment activation fails
```bash
# Recreate virtual environment
rm -rf venv311
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
```

**Problem**: Import errors in Python
```bash
# Verify you're in the virtual environment
which python  # Should show path to venv311/bin/python

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

## Next Steps

Once everything is set up:

1. Read the [API Documentation](./services/api/API_DOCS.md)
2. Upload sample documents to test NLP analysis
3. Explore the NLP pipeline in `services/nlp/`
4. Try the ETL tools for batch processing
5. Check out the test suite in `tests/`

## Getting Help

- Check existing GitHub issues
- Review service-specific READMEs
- Consult team documentation
- Ask in team communication channels

---

**Last Updated**: 2025-10-19
**Tested On**: macOS 14, Ubuntu 22.04, Windows 11 (WSL2)
