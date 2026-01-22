#!/usr/bin/env bash
set -euo pipefail

sd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${sd}/../.." && pwd)"

echo "==> build (serial, no OpenMP)"
(cd "${root}" && make clean && make shud)

echo "==> python unit tests (coverage >= 90%)"
python3 -m coverage run --rcfile=/dev/null --source=validation/baseline -m unittest discover -s "${sd}" -p 'test_*.py'
python3 -m coverage report --rcfile=/dev/null --fail-under=90

echo "==> compare against golden baseline"
python3 "${sd}/compare_baseline.py"

echo "==> OK"
