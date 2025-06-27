#!/bin/bash
# Build script for Karenina documentation

# Set Python path to include source directory
export PYTHONPATH="./src:$PYTHONPATH"

# Build documentation
echo "Building Karenina documentation..."
uv run mkdocs build --clean

echo "Documentation built successfully in site/ directory"
echo "To serve locally: uv run mkdocs serve"