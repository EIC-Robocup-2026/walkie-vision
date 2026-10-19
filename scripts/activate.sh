#!/usr/bin/env bash
# Sourced by pixi on environment activation.

# This repository's message package (available after `pixi run build-msgs`).
# Pixi does not show this script's output (Pixi 0.80), so a missing build
# prints nothing here; the tests say what to run instead.
if [[ -f "${PIXI_PROJECT_ROOT}/ros2_ws/install/local_setup.sh" ]]; then
    # shellcheck source=/dev/null
    source "${PIXI_PROJECT_ROOT}/ros2_ws/install/local_setup.sh"
fi
