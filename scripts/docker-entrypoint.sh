#!/bin/bash
set -e

# Start Ollama in the background
ollama serve &

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do
  echo "Waiting for Ollama API..."
  sleep 1
done

# Pull required models
echo "Pulling required models..."
ollama pull huihui-ai/Llama-3.2-3B-Instruct-abliterated:llama3.2-3b-Instruct-abliterated
ollama pull meta-llama/Llama-3.1-8B:llama3.1-8b

# Execute the command passed to docker
exec "$@"
