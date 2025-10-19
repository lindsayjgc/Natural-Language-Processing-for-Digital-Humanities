#!/bin/bash

# NLP Document Library - Development Stop Script
# Stops both backend and frontend services and cleans up processes

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Function to check if a port is in use
port_in_use() {
    lsof -ti:"$1" >/dev/null 2>&1
}

# Function to kill processes on a specific port
kill_port() {
    local port=$1
    local service=$2

    if port_in_use "$port"; then
        print_info "Killing processes on port $port ($service)..."

        # Get PIDs using the port
        local pids=$(lsof -ti:"$port" 2>/dev/null)

        if [ ! -z "$pids" ]; then
            # Try graceful shutdown first
            echo "$pids" | xargs kill -TERM 2>/dev/null || true

            # Wait a moment for graceful shutdown
            sleep 2

            # Force kill if still running
            if port_in_use "$port"; then
                print_warning "Force killing processes on port $port..."
                echo "$pids" | xargs kill -9 2>/dev/null || true
                sleep 1
            fi

            if ! port_in_use "$port"; then
                print_success "Stopped $service on port $port"
            else
                print_error "Failed to stop $service on port $port"
                return 1
            fi
        fi
    else
        print_info "No processes found on port $port ($service)"
    fi
}

# Function to clean up log files
cleanup_logs() {
    print_info "Cleaning up log files..."

    if [ -f "backend.log" ]; then
        rm -f backend.log
        print_info "Removed backend.log"
    fi

    if [ -f "frontend.log" ]; then
        rm -f frontend.log
        print_info "Removed frontend.log"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -k, --keep     Keep log files (don't delete them)"
    echo "  -v, --verbose  Show verbose output"
    echo ""
    echo "This script stops the NLP Document Library development services"
    echo "and cleans up any running processes on ports 8000 and 3000."
}

# Main execution
main() {
    local keep_logs=false
    local verbose=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -k|--keep)
                keep_logs=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    print_info "Stopping NLP Document Library Development Services"
    echo ""

    # Stop backend (port 8000)
    kill_port 8000 "backend"

    # Stop frontend (port 3000)
    kill_port 3000 "frontend"

    # Additional cleanup - kill any Node.js processes that might be stuck
    print_info "Checking for stuck Node.js processes..."
    local node_pids=$(pgrep -f "next-server\|next dev" 2>/dev/null || true)

    if [ ! -z "$node_pids" ]; then
        print_warning "Found stuck Node.js processes, killing them..."
        echo "$node_pids" | xargs kill -9 2>/dev/null || true
    fi

    # Clean up log files unless --keep flag is used
    if [ "$keep_logs" = false ]; then
        cleanup_logs
    else
        print_info "Keeping log files as requested"
    fi

    echo ""
    print_success "Development services stopped successfully!"

    if [ "$verbose" = true ]; then
        print_info "Verification - checking if ports are free:"
        if ! port_in_use 8000 && ! port_in_use 3000; then
            print_success "All ports are now free"
        else
            print_warning "Some ports may still be in use"
            if port_in_use 8000; then
                print_warning "Port 8000 is still in use"
            fi
            if port_in_use 3000; then
                print_warning "Port 3000 is still in use"
            fi
        fi
    fi
}

# Run main function
main "$@"
