#!/bin/bash
set -e

if [ "$ENV" = "prod" ]; then
    echo "Starting in Production Mode with $WORKERS workers..."
    if [ -z "$WORKERS" ] || [ "$WORKERS" -eq 1 ]; then
        WORKERS=$(python -c "import multiprocessing; print(min(max(multiprocessing.cpu_count(), 1), 2))")
        echo "Auto-calculated workers: $WORKERS"
    fi
    
    exec gunicorn app:app \
        --bind=0.0.0.0:8000 \
        --workers=${WORKERS:-2} \
        --worker-class=${WORKER_CLASS:-uvicorn.workers.UvicornWorker} \
        --worker-connections=${WORKER_CONNECTIONS:-500} \
        --max-requests=${MAX_REQUESTS:-500} \
        --max-requests-jitter=${MAX_REQUESTS_JITTER:-50} \
        --keep-alive=${KEEPALIVE:-2} \
        --timeout=600 \
        --graceful-timeout=120 \
        --worker-tmp-dir=/dev/shm \
        --preload \
        --log-level=info \
        --access-logfile=- \
        --error-logfile=- \
        --capture-output
else
    echo "Starting in Development Mode..."
    exec uvicorn app:app --host=0.0.0.0 --port=8000 --reload --log-level=info
fi 