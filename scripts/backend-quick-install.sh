set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Requirements Check
command -v node >/dev/null 2>&1 || error "Node.js is required"
command -v pnpm >/dev/null 2>&1 || { info "Installing pnpm"; npm install -g pnpm; }
command -v python3 >/dev/null 2>&1 || error "Python 3.11+ is required"

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
[[ "$PYTHON_VERSION" < "3.11" ]] && error "Python 3.11+ required, found $PYTHON_VERSION"
success "System requirements satisfied"

# Backend Setup
info "Setting up backend"

# Python venv
[ ! -d "venv311" ] && python3 -m venv venv311
source venv311/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# NLTK packages
python3 -c "import nltk; [nltk.download(pkg, quiet=True) for pkg in ['punkt_tab','stopwords','wordnet','vader_lexicon','averaged_perceptron_tagger_eng','maxent_ne_chunker','words']]"

# spaCy model
python3 -m spacy download en_core_web_sm

# Node.js dependencies for backend
pnpm install --filter .

success "Backend setup complete"

echo ""
info "Start backend with:"
echo "  source venv311/bin/activate && pnpm --filter . dev -- --port 8000"

