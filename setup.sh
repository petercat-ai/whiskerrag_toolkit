#!/bin/bash
# setup.sh

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

VENV="venv"
if [ ! -d "$VENV" ]; then
    info "Creating virtual environment..."
    python3 -m venv $VENV
    success "Virtual environment created."
fi

info "Activating virtual environment..."
source "$VENV/bin/activate"

info "Upgrading pip..."
pip install --upgrade pip

info "Installing production dependencies..."
pip install -r requirements.txt
pip install -e .

info "Installing development dependencies..."
pip install -r requirements-dev.txt

info "Setting up pre-commit hooks..."
pre-commit install

success "Setup completed successfully!"
echo "To activate the virtual environment, run: source venv/bin/activate"

exec $SHELL
