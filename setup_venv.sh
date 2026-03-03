#!/bin/bash

# Script to initialize a Python virtual environment for BatteryArbitrage

set -e  # Exit on error

VENV_NAME="venv"

echo "Creating virtual environment: $VENV_NAME"
python3 -m venv $VENV_NAME

echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "✓ Virtual environment created and activated!"
echo "✓ Dependencies installed from requirements.txt"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
