#!/usr/bin/env bash
# ==============================================================================
# Launch Script for 3D Gazebo Simulation (Harmonic / ROS2 Jazzy)
# Usage:
#   ./simulation/launch_gazebo.sh [--headless] [--world path]
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WORLD_FILE="${REPO_ROOT}/simulation/worlds/navigation_world.sdf"
MODELS_DIR="${REPO_ROOT}/simulation/models"

HEADLESS=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --headless|-s) HEADLESS=true; shift ;;
        --world) WORLD_FILE="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Check Gazebo Sim executable
GZ_CMD=""
if command -v gz &> /dev/null; then
    GZ_CMD="gz"
elif [ -f "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz" ]; then
    GZ_CMD="/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz"
else
    echo "[ERROR] Gazebo Sim ('gz') not found on system."
    exit 1
fi

# Set resource paths for SDF models and worlds
export GZ_SIM_RESOURCE_PATH="${MODELS_DIR}:${REPO_ROOT}/simulation/worlds:${GZ_SIM_RESOURCE_PATH}"

echo "=========================================================="
echo " Starting 3D Gazebo Simulation"
echo " World:  ${WORLD_FILE}"
echo " Models: ${MODELS_DIR}"
echo " Mode:   $([ "$HEADLESS" = true ] && echo "Headless (Server)" || echo "Full GUI")"
echo "=========================================================="

if [ "$HEADLESS" = true ]; then
    exec "${GZ_CMD}" sim -s -r "${WORLD_FILE}"
else
    exec "${GZ_CMD}" sim -r "${WORLD_FILE}"
fi
