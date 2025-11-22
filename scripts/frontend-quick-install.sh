set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

info "Setting up frontend"

cd services/frontend

# Install frontend dependencies
pnpm install

success "Frontend setup complete"

echo ""
info "Start frontend with:"
echo "PORT=3000 pnpm dev --hostname 0.0.0.0"
