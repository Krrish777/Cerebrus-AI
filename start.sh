#!/bin/bash
# Cerebrus AI Startup Script for Cloud Run / Dockerless Deployment

# Set environment variables
export PORT=${PORT:-8080}
export HOST=${HOST:-0.0.0.0}

# Log startup information
echo "Starting Cerebrus AI on $HOST:$PORT"

# Start Streamlit with proper configuration
streamlit run main.py \
  --server.port=$PORT \
  --server.address=$HOST \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --browser.gatherUsageStats=false
