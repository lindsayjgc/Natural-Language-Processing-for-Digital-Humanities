#!/bin/bash
# Non-interactive backend startup for Fly.io

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

info() { echo -e "${BLUE}[BACKEND]${NC} $1"; }
success() { echo -e "${GREEN}[BACKEND]${NC} $1"; }

# Activate Python venv
source venv311/bin/activate

# Ensure backend directory exists
if [ ! -d "services/api" ]; then
    echo "[ERROR] services/api folder not found!"
    exit 1
fi

info "Starting backend API server on port 8000..."
cd services/api
uvicorn api:app --host 0.0.0.0 --port 8000
