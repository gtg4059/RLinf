#!/usr/bin/env bash
# Bake the current container's Isaac Lab install into a local image tag, OR rebuild
# the official embodied-isaaclab stage (pinned Isaac Lab 2.3.0).
#
# Run on the HOST (needs docker CLI), not inside this container.
#
# Usage:
#   # A) Rebuild from Dockerfile (recommended, reproducible):
#   bash docker/bake_isaaclab_into_image.sh rebuild
#
#   # B) Commit a running container that already has /opt/envs/isaaclab:
#   bash docker/bake_isaaclab_into_image.sh commit [container_name_or_id]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-rebuild}"
IMAGE_TAG="${IMAGE_TAG:-rlinf:embodied-isaaclab-blackwell}"

case "$MODE" in
  rebuild)
    exec bash "$REPO_ROOT/docker/build_embodied_isaaclab_blackwell.sh" --tag "$IMAGE_TAG"
    ;;
  commit)
    CID="${2:-}"
    if [ -z "$CID" ]; then
      # Prefer the well-known run helper name, else the first running rlinf container.
      if docker ps --format '{{.Names}}' | grep -qx 'rlinf-isaaclab-blackwell'; then
        CID=rlinf-isaaclab-blackwell
      else
        CID="$(docker ps --format '{{.ID}}\t{{.Names}}\t{{.Image}}' | awk '/rlinf|isaaclab|openpi/ {print $1; exit}')"
      fi
    fi
    if [ -z "$CID" ]; then
      echo "ERROR: no container id/name given and none auto-detected." >&2
      echo "Usage: $0 commit <container_name_or_id>" >&2
      exit 1
    fi
    echo "[bake] docker commit $CID -> $IMAGE_TAG"
    docker commit \
      --change 'ENV ISAAC_LAB_PATH=/opt/envs/isaaclab' \
      --change 'ENV ISAAC_LAB_VERSION=2.3.0' \
      --change 'ENV ISAAC_LAB_GIT_REF=4246b6b4f4a3e74ee20e002ed7536b1c788d39f4' \
      --change 'LABEL rlinf.isaaclab_version=2.3.0' \
      "$CID" "$IMAGE_TAG"
    echo "[bake] done: $IMAGE_TAG"
    docker image inspect "$IMAGE_TAG" --format '{{.Id}} {{.Config.Env}}' | head -c 500
    echo
    ;;
  *)
    echo "Usage: $0 {rebuild|commit} [container]" >&2
    exit 1
    ;;
esac
