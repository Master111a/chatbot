#!/bin/bash

echo "Initializing Ollama models..."

while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    echo "Waiting for Ollama to start..."
    sleep 2
done

echo "Ollama is ready!"

echo "Pulling models..."
ollama pull qwen3:1.7b
ollama pull qwen3:8b

echo "Warming up models..."

curl -s -X POST http://localhost:11434/api/generate -d '{
    "model": "qwen3:1.7b",
    "prompt": "Hello",
    "stream": false,
    "keep_alive": -1,
    "options": {"num_predict": 1}
}' > /dev/null

echo "Qwen3:1.7b warmed up"

curl -s -X POST http://localhost:11434/api/generate -d '{
    "model": "qwen3:8b",
    "prompt": "Hello", 
    "stream": false,
    "keep_alive": -1,
    "options": {"num_predict": 1}
}' > /dev/null

echo "Qwen3:8b warmed up"

echo "All models initialized and loaded into memory!" 