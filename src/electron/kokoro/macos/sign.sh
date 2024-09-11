#!/bin/bash

# Fail on any error.
set -e

# The tar file produced by the macos_external job.
ELECTION_APP_TAR_FILE="${KOKORO_ARTIFACTS_CreateDIR}/artifacts/app.tar.gz"
ELECTRON_BASE_DIR="${KOKORO_ARTIFACTS_DIR}/github/model-explorer/src/electron"

# Replace the whole electron app.
echo
echo '#### Replace electron app'
rm -rf "${ELECTRON_BASE_DIR}/app"

echo
echo '---------- before'
ls -lh ${ELECTRON_BASE_DIR}
echo

cd "${ELECTRON_BASE_DIR}"
tar -xf $ELECTION_APP_TAR_FILE

echo
echo '---------- after'
ls -lh ${ELECTRON_BASE_DIR}
echo
