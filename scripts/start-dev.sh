#!/bin/bash

# NLP Document Library - Development Startup Script
# Starts both backend (FastAPI) and frontend (Next.js) services

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

print_backend() {
    echo -e "${PURPLE}[BACKEND]${NC} $1"
}

print_frontend() {
    echo -e "${CYAN}[FRONTEND]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -ti:"$1" >/dev/null 2>&1
}

# Function to kill processes on specific ports
kill_port() {
    local port=$1
    local service=$2

    if port_in_use "$port"; then
        print_warning "Port $port is already in use. Killing existing $service process..."
        lsof -ti:"$port" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

# Function to cleanup background processes
cleanup() {
    print_info "Shutting down services..."

    if [ ! -z "$BACKEND_PID" ]; then
        print_backend "Stopping backend (PID: $BACKEND_PID)"
        kill $BACKEND_PID 2>/dev/null || true
    fi

    if [ ! -z "$FRONTEND_PID" ]; then
        print_frontend "Stopping frontend (PID: $FRONTEND_PID)"
        kill $FRONTEND_PID 2>/dev/null || true
    fi

    # Additional cleanup - kill any remaining processes on our ports
    kill_port 8000 "backend"
    kill_port 3000 "frontend"

    print_success "Services stopped successfully"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Function to check if installation is complete
check_installation() {
    local missing_items=()

    # Check virtual environment
    if [ ! -d "venv311" ]; then
        missing_items+=("Python virtual environment")
    fi

    # Check frontend dependencies
    if [ ! -d "services/frontend/node_modules" ]; then
        missing_items+=("Frontend dependencies")
    fi

    # Check environment files
    if [ ! -f ".env" ]; then
        missing_items+=("Backend environment file (.env)")
    fi

    if [ ! -f "services/frontend/.env.local" ]; then
        missing_items+=("Frontend environment file (.env.local)")
    fi

    if [ ${#missing_items[@]} -gt 0 ]; then
        return 1
    fi

    return 0
}

# Main execution
main() {
    local auto_install=false
    local force_install=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --auto-install)
                auto_install=true
                shift
                ;;
            --install)
                force_install=true
                shift
                ;;
            *)
                shift
                ;;
        esac
    done

    print_info "Starting NLP Document Library Development Environment"
    echo ""

    # Check if we're in the right directory
    if [ ! -f "requirements.txt" ] || [ ! -d "services" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi

    # Check installation status
    if ! check_installation; then
        print_warning "Installation appears incomplete. Missing components detected."
        echo ""

        if [ "$force_install" = true ]; then
            print_info "Running installation script..."
            ./install.sh --yes
        elif [ "$auto_install" = true ]; then
            print_info "Auto-installing missing components..."
            ./install.sh --yes
        else
            print_info "To complete the installation, run:"
            echo "  ./install.sh"
            echo ""
            print_info "Or to auto-install and start:"
            echo "  ./start-dev.sh --install"
            echo ""
            read -p "Run installation now? [Y/n]: " run_install

            if [[ ! "$run_install" =~ ^[Nn]$ ]]; then
                print_info "Running installation script..."
                ./install.sh
            else
                print_error "Installation required to continue"
                exit 1
            fi
        fi
    fi

    # Check prerequisites
    print_info "Checking prerequisites..."

    # Check Python virtual environment
    if [ ! -d "venv311" ]; then
        print_error "Virtual environment 'venv311' not found. Please run: ./install.sh"
        exit 1
    fi

    # Check Node.js and pnpm
    if ! command_exists node; then
        print_error "Node.js not found. Please install Node.js 18+ first"
        exit 1
    fi

    if ! command_exists pnpm; then
        print_error "pnpm not found. Please install pnpm first:"
        echo "  npm install -g pnpm"
        exit 1
    fi

    # Check frontend dependencies
    if [ ! -d "services/frontend/node_modules" ]; then
        print_warning "Frontend dependencies not found. Installing..."
        cd services/frontend
        pnpm install
        cd ../..
    fi

    # Check environment file
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Please create one from env.example:"
        echo "  cp env.example .env"
        echo "  # Then edit .env with your MongoDB connection string"
        echo ""
    fi

    print_success "Prerequisites check passed"
    echo ""

    # Kill any existing processes on our ports
    kill_port 8000 "backend"
    kill_port 3000 "frontend"

    # Start backend
    print_info "Starting backend API server on port 8000..."
    cd services/api
    ../../venv311/bin/uvicorn api:app --reload --port 8000 --host 0.0.0.0 > ../../backend.log 2>&1 &
    BACKEND_PID=$!
    cd ../..

    # Wait a moment for backend to start
    sleep 3

    # Check if backend started successfully
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        print_error "Backend failed to start. Check backend.log for details:"
        cat backend.log
        exit 1
    fi

    # Start frontend
    print_info "Starting frontend development server on port 3000..."
    cd services/frontend

    # Ensure frontend environment file exists
    if [ ! -f ".env.local" ]; then
        echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
    fi

    pnpm dev > ../../frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ../..

    # Wait a moment for frontend to start
    sleep 5

    # Check if frontend started successfully
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        print_error "Frontend failed to start. Check frontend.log for details:"
        cat frontend.log
        cleanup
        exit 1
    fi

    print_success "Both services started successfully!"
    echo ""
    print_info "Access points:"
    print_info "  Frontend: http://localhost:3000"
    print_info "  Backend API: http://localhost:8000"
    print_info "  API Documentation: http://localhost:8000/docs"
    echo ""
    print_info "Logs are being written to:"
    print_info "  Backend: backend.log"
    print_info "  Frontend: frontend.log"
    echo ""
    print_warning "Press Ctrl+C to stop both services"
    echo ""

    # Monitor and display logs
    tail -f backend.log | sed "s/^/$(echo -e ${PURPLE}[BACKEND]${NC}) /" &
    tail -f frontend.log | sed "s/^/$(echo -e ${CYAN}[FRONTEND]${NC}) /" &

    # Wait for either process to exit
    wait $BACKEND_PID $FRONTEND_PID
}

# Run main function
main "$@"
