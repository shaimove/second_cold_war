#!/usr/bin/env bash
# Convenience launcher for local development.
# Usage: bash run_local.sh
set -euo pipefail

if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
  python3 -m venv .venv
fi

# Prefer .venv, fall back to venv if it already exists.
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

pip install --upgrade pip >/dev/null
pip install -r requirements.txt

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add OPENAI_API_KEY to enable live mode."
fi

python scripts/ingest_docs.py || true

uvicorn app.main:app --reload --port 8000
