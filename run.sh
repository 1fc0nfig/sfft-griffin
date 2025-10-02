#!/bin/bash
# Simple helper to run the converter and open a spectrogram preview.

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 <image> [converter-flags...]" >&2
    echo "Example: $0 /examples/input.jpg --duration 6 --iterations 64" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/" && pwd)"

INPUT_FILE="$1"
shift
CARGO_ARGS="$@"

BASENAME="$(basename "${INPUT_FILE}")"
OUTPUT_FILE="${BASENAME%.*}.wav"

LOG_FREQ=""
if [[ " ${CARGO_ARGS} " == *" --log-freq "* ]]; then
    LOG_FREQ="--log-freq"
fi

pushd "${PROJECT_ROOT}" >/dev/null

# Check if we're in a virtual environment
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "🚩 Error: This script must be run within a Python virtual environment." >&2
    echo "Please create and activate the virtual environment first:" >&2
    echo "  python3 -m venv .venv" >&2
    echo "  source .venv/bin/activate" >&2
    echo "Then run the script again:" >&2
    echo "  source venv/bin/activate" >&2
    echo "  $0 $@" >&2
    exit 1
fi

echo "Converting ${INPUT_FILE} → ${OUTPUT_FILE}"
if [ -n "${CARGO_ARGS}" ]; then
    cargo run --release -- -i "${INPUT_FILE}" -o "${OUTPUT_FILE}" ${CARGO_ARGS}
else
    cargo run --release -- -i "${INPUT_FILE}" -o "${OUTPUT_FILE}"
fi

echo "Rendering spectrogram preview for ${OUTPUT_FILE}"
PYTHON="python3"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    PYTHON="python"
fi

"${PYTHON}" "${SCRIPT_DIR}/scripts/audio_histogram.py" "${OUTPUT_FILE}" --only-spectrogram ${LOG_FREQ}

popd >/dev/null
