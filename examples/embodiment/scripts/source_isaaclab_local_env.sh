# Source machine-local IsaacLab paths if the gitignored env file exists.
# Usage: source this file, then source_isaaclab_local_env <repo_root>
#
# Also aliases ISAAC_SIM_PATH <-> ISAAC_PATH so the Docker wrapper and
# run_embodiment.sh see the same tree.

source_isaaclab_local_env() {
  local repo_root="${1:-}"
  local env_file
  if [ -z "${repo_root}" ]; then
    echo "source_isaaclab_local_env: repo root is not set" >&2
    return 1
  fi
  env_file="${repo_root}/examples/embodiment/scripts/isaaclab_local.env"
  if [ -f "${env_file}" ]; then
    # shellcheck disable=SC1090
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
  # Only alias a path that is actually an Isaac Sim tree. A host path from
  # isaaclab_local.env is often invisible inside Docker; leave it unset so
  # run_embodiment.sh can probe /workspace/RLinf or /workspace/isaac_sim.
  if [ -n "${ISAAC_SIM_PATH:-}" ] && [ -z "${ISAAC_PATH:-}" ] \
      && [ -f "${ISAAC_SIM_PATH}/setup_conda_env.sh" ]; then
    export ISAAC_PATH="${ISAAC_SIM_PATH}"
  fi
  if [ -n "${ISAAC_PATH:-}" ] && [ -z "${ISAAC_SIM_PATH:-}" ] \
      && [ -f "${ISAAC_PATH}/setup_conda_env.sh" ]; then
    export ISAAC_SIM_PATH="${ISAAC_PATH}"
  fi
}
