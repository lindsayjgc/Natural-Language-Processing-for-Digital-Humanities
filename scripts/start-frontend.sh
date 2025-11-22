set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

info() { echo -e "${BLUE}[FRONTEND]${NC} $1"; }
success() { echo -e "${GREEN}[FRONTEND]${NC} $1"; }

# Ensure frontend dependencies are installed
cd services/frontend
if [ ! -d "node_modules" ]; then
    info "Installing frontend dependencies..."
    pnpm install
fi

# Ensure environment file exists
if [ ! -f ".env.local" ]; then
    echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
fi

info "Starting frontend on port 3000..."
PORT=3000 pnpm dev --hostname 0.0.0.0
