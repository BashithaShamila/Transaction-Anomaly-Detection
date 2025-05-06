#!/bin/bash

# Default to running the fraud pipeline with sample data
if [ "$#" -eq 0 ]; then
    echo "Running fraud detection pipeline with sample data..."
    python /app/src/fraud_pipeline.py
    exit 0
fi

# Handle specific commands
case "$1" in
    --isolation)
        echo "Running isolation forest detector only..."
        conda run -n isolation_env python /app/src/isolation_forest_detector.py
        ;;
        
    --autoencoder)
        echo "Running autoencoder detector only..."
        conda run -n autoencoder_env python /app/src/autoencoder_detector.py
        ;;
        
    --pipeline)
        echo "Running full fraud detection pipeline..."
        if [ "$#" -eq 3 ] && [ "$2" == "--input" ]; then
            python /app/src/fraud_pipeline.py --input "$3"
        else
            python /app/src/fraud_pipeline.py
        fi
        ;;
        
    --help)
        echo "Fraud Detection Pipeline"
        echo "Usage:"
        echo "  ./run_pipeline.sh                     Run pipeline with sample data"
        echo "  ./run_pipeline.sh --isolation         Run isolation forest detector only"
        echo "  ./run_pipeline.sh --autoencoder       Run autoencoder detector only"
        echo "  ./run_pipeline.sh --pipeline          Run full pipeline with sample data"
        echo "  ./run_pipeline.sh --pipeline --input [file.json]  Run pipeline with custom input"
        ;;
        
    *)
        echo "Unknown command: $1"
        echo "Run ./run_pipeline.sh --help for usage information"
        exit 1
        ;;
esac