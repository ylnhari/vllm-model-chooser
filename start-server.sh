#!/bin/bash

# vLLM Model Chooser - Development Server
# Usage: ./start-server.sh [port]
# If port is not specified, uses 5000. If port is in use, suggests alternatives.

DEFAULT_PORT=5000
PORT=${1:-$DEFAULT_PORT}

# Function to check if port is in use
check_port() {
    lsof -i :$1 >/dev/null 2>&1
}

# Function to find next available port
find_available_port() {
    local start=$1
    local port=$start
    while check_port $port; do
        port=$((port + 1))
        if [ $port -gt 65535 ]; then
            echo "Error: No available ports found"
            exit 1
        fi
    done
    echo $port
}

# Check if specified port is available
if check_port $PORT; then
    echo "Port $PORT is already in use."
    echo ""
    echo "Suggested free ports:"
    for p in 3000 5001 5002 8080 8888 3001 3002; do
        if ! check_port $p; then
            echo "  - $p"
        fi
    done
    echo ""
    read -p "Would you like me to find the next available port? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        AVAILABLE=$(find_available_port $((PORT + 1)))
        echo "Starting server on port $AVAILABLE..."
        PORT=$AVAILABLE
    else
        echo "Server not started. Run with a different port: ./start-server.sh [port]"
        exit 1
    fi
fi

echo "Starting vLLM Model Chooser on http://localhost:$PORT"
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
python3 -m http.server $PORT