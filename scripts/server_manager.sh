#!/bin/bash

# Server management script with automatic PID handling
PID_FILE="/tmp/vllm_api_server.pid"
LOG_FILE="/tmp/vllm_api_server.log"

start_server() {
    echo "========================================="
    echo "Starting vLLM API Server"
    echo "========================================="

    # Check if server is already running
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat $PID_FILE)
        if ps -p $OLD_PID > /dev/null 2>&1; then
            echo "❌ Server already running with PID: $OLD_PID"
            echo "   Stop it first with: ./server_manager.sh stop"
            exit 1
        else
            echo "⚠️  Removing stale PID file"
            rm $PID_FILE
        fi
    fi

    # Start the server
    echo "Starting server..."
    python src/api_server.py > $LOG_FILE 2>&1 &
    SERVER_PID=$!

    # Save PID to file
    echo $SERVER_PID > $PID_FILE

    # Wait a moment for server to start
    sleep 3

    # Check if server started successfully
    if ps -p $SERVER_PID > /dev/null; then
        # Test health endpoint
        curl -s http://localhost:8000/health > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✅ Server started successfully!"
            echo "   PID: $SERVER_PID (saved to $PID_FILE)"
            echo "   Log: $LOG_FILE"
            echo ""
            echo "   Stop with: ./server_manager.sh stop"
            echo "   Status: ./server_manager.sh status"
            echo "   Logs: ./server_manager.sh logs"
        else
            echo "⚠️  Server started but health check failed"
            echo "   Check logs: tail -f $LOG_FILE"
        fi
    else
        echo "❌ Failed to start server"
        echo "   Check logs: cat $LOG_FILE"
        rm -f $PID_FILE
        exit 1
    fi
}

stop_server() {
    echo "========================================="
    echo "Stopping vLLM API Server"
    echo "========================================="

    if [ -f "$PID_FILE" ]; then
        PID=$(cat $PID_FILE)
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID
            echo "✅ Server stopped (PID: $PID)"
            rm -f $PID_FILE
        else
            echo "⚠️  Server not running (stale PID: $PID)"
            rm -f $PID_FILE
        fi
    else
        echo "❌ No PID file found. Server may not be running."
        echo "   Check manually: ps aux | grep api_server"
    fi
}

restart_server() {
    echo "========================================="
    echo "Restarting vLLM API Server"
    echo "========================================="
    stop_server
    echo ""
    sleep 2
    start_server
}

server_status() {
    echo "========================================="
    echo "vLLM API Server Status"
    echo "========================================="

    if [ -f "$PID_FILE" ]; then
        PID=$(cat $PID_FILE)
        if ps -p $PID > /dev/null 2>&1; then
            echo "✅ Server is running"
            echo "   PID: $PID"
            echo "   Uptime: $(ps -o etime= -p $PID | xargs)"

            # Check health endpoint
            HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null)
            if [ $? -eq 0 ]; then
                echo "   Health: OK"
                echo ""
                echo "   Endpoints:"
                echo "     http://localhost:8000/health"
                echo "     http://localhost:8000/models"
                echo "     http://localhost:8000/v1/generate"
                echo "     http://localhost:8000/metrics"
                echo "     http://localhost:8000/docs"
            else
                echo "   Health: ⚠️  Not responding"
            fi
        else
            echo "❌ Server not running (stale PID: $PID)"
            rm -f $PID_FILE
        fi
    else
        echo "❌ Server not running (no PID file)"
    fi

    echo ""
    echo "Log file: $LOG_FILE"
}

show_logs() {
    echo "========================================="
    echo "vLLM API Server Logs"
    echo "========================================="

    if [ -f "$LOG_FILE" ]; then
        echo "Showing last 20 lines (use 'tail -f $LOG_FILE' for live logs):"
        echo "-----------------------------------------"
        tail -20 $LOG_FILE
    else
        echo "❌ No log file found"
    fi
}

test_quick() {
    echo "========================================="
    echo "Quick API Test"
    echo "========================================="

    # Check if server is running
    if [ ! -f "$PID_FILE" ]; then
        echo "❌ Server not running. Start with: ./server_manager.sh start"
        exit 1
    fi

    echo "1. Health Check:"
    curl -s http://localhost:8000/health | python -m json.tool | head -5

    echo ""
    echo "2. Model List:"
    curl -s http://localhost:8000/models | python -m json.tool | grep name

    echo ""
    echo "3. Simple Inference:"
    curl -X POST http://localhost:8000/v1/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Hello, how are you?", "user_id": "test"}' \
        -s | python -m json.tool | grep -E "(complexity|model_used|response)" | head -3

    echo ""
    echo "✅ Quick test complete"
}

# Main command handler
case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        server_status
        ;;
    logs)
        show_logs
        ;;
    test)
        test_quick
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|test}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the API server"
        echo "  stop    - Stop the API server"
        echo "  restart - Restart the API server"
        echo "  status  - Check server status"
        echo "  logs    - Show server logs"
        echo "  test    - Run quick API test"
        echo ""
        echo "Server PID saved to: $PID_FILE"
        echo "Server logs saved to: $LOG_FILE"
        ;;
esac