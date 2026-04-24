#!/bin/bash

# Set the directory containing example kernel files
EXAMPLE_DIR="examples"
OUTPUT_DIR="optimized_outputs"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Iterate over each file in the example directory
for kernel_file in "$EXAMPLE_DIR"/*.py; do
    if [ -f "$kernel_file" ]; then
        # Extract the base filename without extension
        base_name=$(basename "$kernel_file" .py)

        # Run the optimization pipeline for the current kernel file
        echo "Processing $kernel_file..."
        python3 run_client.py "$kernel_file" -o "$OUTPUT_DIR/$base_name.py"
    fi
done

echo "All example kernels have been processed. Optimized files are in $OUTPUT_DIR."