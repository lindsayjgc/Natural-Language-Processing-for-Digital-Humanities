#!/bin/bash

# NLP Document Library - Installation Script
# Complete setup for the NLP Document Library development environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

print_header() {
    echo -e "${BOLD}${CYAN}$1${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    local python_cmd=$1
    if command_exists "$python_cmd"; then
        local version=$($python_cmd --version 2>&1 | cut -d' ' -f2)
        local major=$(echo $version | cut -d'.' -f1)
        local minor=$(echo $version | cut -d'.' -f2)

        if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
            return 0
        fi
    fi
    return 1
}

# Function to prompt user for input
prompt_user() {
    local prompt=$1
    local default=$2
    local response

    if [ -n "$default" ]; then
        read -p "$prompt [$default]: " response
        response=${response:-$default}
    else
        read -p "$prompt: " response
    fi

    echo "$response"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -y, --yes         Skip all prompts (use defaults)"
    echo "  --skip-mongodb    Skip MongoDB Atlas setup"
    echo "  --skip-nltk       Skip NLTK data download"
    echo "  --skip-spacy      Skip spaCy model download"
    echo ""
    echo "This script will set up the complete development environment for"
    echo "the NLP Document Library including Python, Node.js, and database setup."
}

# Main installation function
main() {
    local auto_yes=false
    local skip_mongodb=false
    local skip_nltk=false
    local skip_spacy=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -y|--yes)
                auto_yes=true
                shift
                ;;
            --skip-mongodb)
                skip_mongodb=true
                shift
                ;;
            --skip-nltk)
                skip_nltk=true
                shift
                ;;
            --skip-spacy)
                skip_spacy=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    print_header "🚀 NLP Document Library - Installation Script"
    echo ""
    print_info "This script will set up your complete development environment."
    echo ""

    # Check if we're in the right directory
    if [ ! -f "requirements.txt" ] || [ ! -d "services" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi

    # Step 1: System Requirements Check
    print_step "1/8 Checking system requirements..."

    # Check Python 3.11
    print_info "Checking Python 3.11..."
    if check_python_version "python3.11"; then
        print_success "Python 3.11 found"
        PYTHON_CMD="python3.11"
    elif check_python_version "python3"; then
        local version=$(python3 --version 2>&1 | cut -d' ' -f2)
        local major=$(echo $version | cut -d'.' -f1)
        local minor=$(echo $version | cut -d'.' -f2)

        if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
            print_success "Python $version found (compatible)"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.11+ required. Found: $version"
            print_info "Please install Python 3.11+ from https://python.org"
            exit 1
        fi
    else
        print_error "Python not found. Please install Python 3.11+"
        exit 1
    fi

    # Check Node.js
    print_info "Checking Node.js..."
    if command_exists node; then
        local node_version=$(node --version)
        print_success "Node.js $node_version found"
    else
        print_error "Node.js not found. Please install Node.js 18+ from https://nodejs.org"
        exit 1
    fi

    # Check pnpm
    print_info "Checking pnpm..."
    if command_exists pnpm; then
        local pnpm_version=$(pnpm --version)
        print_success "pnpm $pnpm_version found"
    else
        print_warning "pnpm not found. Installing pnpm..."
        if [ "$auto_yes" = true ]; then
            npm install -g pnpm
        else
            read -p "Install pnpm globally? [Y/n]: " install_pnpm
            if [[ "$install_pnpm" =~ ^[Nn]$ ]]; then
                print_error "pnpm is required. Please install it manually: npm install -g pnpm"
                exit 1
            else
                npm install -g pnpm
            fi
        fi
        print_success "pnpm installed"
    fi

    # Step 2: Python Virtual Environment
    print_step "2/8 Setting up Python virtual environment..."

    if [ -d "venv311" ]; then
        print_warning "Virtual environment 'venv311' already exists"
        if [ "$auto_yes" = false ]; then
            read -p "Recreate virtual environment? [y/N]: " recreate_venv
            if [[ "$recreate_venv" =~ ^[Yy]$ ]]; then
                print_info "Removing existing virtual environment..."
                rm -rf venv311
            fi
        fi
    fi

    if [ ! -d "venv311" ]; then
        print_info "Creating virtual environment..."
        $PYTHON_CMD -m venv venv311
        print_success "Virtual environment created"
    fi

    # Step 3: Install Python Dependencies
    print_step "3/8 Installing Python dependencies..."

    print_info "Activating virtual environment and upgrading pip..."
    source venv311/bin/activate
    pip install --upgrade pip

    print_info "Installing Python packages from requirements.txt..."
    pip install -r requirements.txt

    print_success "Python dependencies installed"

    # Step 4: Download NLTK Data
    if [ "$skip_nltk" = false ]; then
        print_step "4/8 Downloading NLTK data packages..."

        print_info "Downloading required NLTK packages..."
        python -c "
import nltk
packages = [
    'punkt_tab',
    'stopwords',
    'wordnet',
    'vader_lexicon',
    'averaged_perceptron_tagger_eng',
    'maxent_ne_chunker',
    'words'
]
for package in packages:
    try:
        nltk.download(package, quiet=True)
        print(f'Downloaded {package}')
    except Exception as e:
        print(f'Warning: Failed to download {package}: {e}')
"
        print_success "NLTK data packages downloaded"
    else
        print_step "4/8 Skipping NLTK data download"
    fi

    # Step 5: Download spaCy Model
    if [ "$skip_spacy" = false ]; then
        print_step "5/8 Downloading spaCy language model..."

        print_info "Downloading en_core_web_sm model..."
        python -m spacy download en_core_web_sm

        print_success "spaCy model downloaded"
    else
        print_step "5/8 Skipping spaCy model download"
    fi

    # Step 6: Install Project Dependencies
    print_step "6/8 Installing project dependencies..."

    # Install root workspace dependencies (includes concurrently for pnpm dev)
    print_info "Installing root workspace dependencies..."
    pnpm install --filter .
    print_success "Root dependencies installed"

    # Install frontend dependencies
    print_info "Installing frontend dependencies..."
    cd services/frontend
    pnpm install
    cd ../..
    print_success "Frontend dependencies installed"

    # Step 7: Environment Configuration
    print_step "7/8 Setting up environment files..."

    # Backend environment file
    if [ ! -f ".env" ]; then
        print_info "Creating .env file from template..."
        cp env.example .env

        print_warning "⚠️  IMPORTANT: You need to configure MongoDB Atlas connection"
        echo ""
        print_info "1. Create a free MongoDB Atlas account: https://www.mongodb.com/cloud/atlas"
        print_info "2. Create a cluster and database user"
        print_info "3. Get your connection string"
        print_info "4. Edit .env file and add your MONGODB_URI"
        echo ""

        if [ "$skip_mongodb" = false ]; then
            if [ "$auto_yes" = false ]; then
                read -p "Open .env file for editing now? [Y/n]: " edit_env
                if [[ ! "$edit_env" =~ ^[Nn]$ ]]; then
                    ${EDITOR:-nano} .env
                fi
            fi
        fi
    else
        print_info ".env file already exists"
    fi

    # Frontend environment file
    if [ ! -f "services/frontend/.env.local" ]; then
        print_info "Creating frontend environment file..."
        echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > services/frontend/.env.local
        print_success "Frontend environment file created"
    else
        print_info "Frontend environment file already exists"
    fi

    # Step 8: Verification
    print_step "8/8 Verifying installation..."

    print_info "Testing Python environment..."
    source venv311/bin/activate
    python -c "import pandas, numpy, nltk, spacy, transformers, torch; print('All Python packages imported successfully')"

    print_info "Testing Node.js environment..."
    cd services/frontend
    pnpm --version > /dev/null
    cd ../..

    print_success "Installation verification completed"

    # Final instructions
    echo ""
    print_header "🎉 Installation Complete!"
    echo ""
    print_success "Your NLP Document Library development environment is ready!"
    echo ""
    print_info "Next steps:"
    print_info "1. Configure MongoDB Atlas in .env file (if not done already)"
    print_info "2. Start the development servers:"
    echo "   ./start-dev.sh"
    echo ""
    print_info "Access points:"
    print_info "  Frontend: http://localhost:3000"
    print_info "  Backend API: http://localhost:8000"
    print_info "  API Docs: http://localhost:8000/docs"
    echo ""
    print_info "For detailed setup instructions, see SETUP.md"
    echo ""
}

# Run main function
main "$@"
