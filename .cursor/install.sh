#!/usr/bin/env bash
set -euo pipefail

# Idempotent bootstrap: verify the repository checkout is present and readable.
test -f ReadME
grep -q 'Hello' ReadME
