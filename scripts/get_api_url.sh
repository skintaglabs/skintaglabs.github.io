#!/bin/bash
# Reads API URL from git notes on first commit (same as GH Pages deployment)

git fetch origin refs/notes/commits:refs/notes/commits 2>/dev/null || true
FIRST_COMMIT=$(git rev-list --max-parents=0 HEAD)
API_URL=$(git notes show $FIRST_COMMIT 2>/dev/null || echo "")

if [ -n "$API_URL" ]; then
  echo "$API_URL"
else
  echo "http://localhost:8000"
fi
