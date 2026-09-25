#!/usr/bin/env bash
# Ortam kurulumu (klasör kökünden):  bash kurulum.sh
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import gurobipy; print('gurobipy', gurobipy.gurobi.version())"
echo "Kurulum tamam."
