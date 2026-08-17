#!/usr/bin/env bash
# Migrate Docker data-root from /var/lib/docker to /mnt/E/docker
# Usage: sudo bash toolkits/migrate_docker_to_E.sh
set -euo pipefail

E_UUID="f93a21d2-d5f7-4eb7-a68d-d06e45e3befd"
E_DEV="/dev/disk/by-uuid/${E_UUID}"
E_MNT="/mnt/E"
DOCKER_NEW="${E_MNT}/docker"
DOCKER_OLD="/var/lib/docker"
DAEMON_JSON="/etc/docker/daemon.json"
FSTAB="/etc/fstab"
AUTO_MOUNT="/media/safetics/E"

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: root 권한이 필요합니다. 다음으로 실행하세요:"
  echo "  sudo bash $0"
  exit 1
fi

echo "==> [1/7] E 드라이브 고정 마운트 준비 (${E_MNT})"
if [[ ! -e "${E_DEV}" ]]; then
  echo "ERROR: ${E_DEV} 를 찾을 수 없습니다."
  exit 1
fi

mkdir -p "${E_MNT}"

# 자동 마운트(/media/...)가 있으면 해제 후 /mnt/E 로 재마운트
if mountpoint -q "${AUTO_MOUNT}"; then
  echo "    기존 자동 마운트 해제: ${AUTO_MOUNT}"
  # 사용 중이면 lazy umount
  if ! umount "${AUTO_MOUNT}" 2>/dev/null; then
    echo "    일반 umount 실패 → lazy umount 시도"
    umount -l "${AUTO_MOUNT}"
  fi
fi

if mountpoint -q "${E_MNT}"; then
  echo "    이미 마운트됨: ${E_MNT}"
else
  mount -t ext4 -o defaults,nofail "${E_DEV}" "${E_MNT}"
  echo "    마운트 완료: ${E_MNT}"
fi

# fstab에 없으면 추가 (재부팅 후에도 유지)
FSTAB_LINE="UUID=${E_UUID} ${E_MNT} ext4 defaults,nofail 0 2"
if ! grep -qE "^UUID=${E_UUID}[[:space:]]+${E_MNT}[[:space:]]" "${FSTAB}"; then
  # 이전 /media 경로 항목이 있으면 주석 처리하지 않고, /mnt/E 항목만 추가
  cp -a "${FSTAB}" "${FSTAB}.bak.$(date +%Y%m%d%H%M%S)"
  echo "${FSTAB_LINE}" >> "${FSTAB}"
  echo "    fstab 항목 추가됨"
else
  echo "    fstab 항목 이미 존재"
fi

df -h "${E_MNT}"

echo "==> [2/7] Docker / containerd 중지"
systemctl stop docker.socket docker containerd || true
# 잔여 프로세스 대기
sleep 2
if pgrep -x dockerd >/dev/null 2>&1; then
  echo "ERROR: dockerd 가 아직 실행 중입니다. 수동으로 중지 후 다시 실행하세요."
  exit 1
fi

echo "==> [3/7] 데이터 복사: ${DOCKER_OLD} → ${DOCKER_NEW}"
mkdir -p "${DOCKER_NEW}"
if [[ -d "${DOCKER_OLD}" ]] && [[ "$(ls -A "${DOCKER_OLD}" 2>/dev/null || true)" ]]; then
  rsync -aHAX --info=progress2 "${DOCKER_OLD}/" "${DOCKER_NEW}/"
else
  echo "WARNING: ${DOCKER_OLD} 가 비어 있거나 없습니다. 새 data-root 만 설정합니다."
fi

echo "==> [4/7] daemon.json 에 data-root 설정"
if [[ -f "${DAEMON_JSON}" ]]; then
  cp -a "${DAEMON_JSON}" "${DAEMON_JSON}.bak.$(date +%Y%m%d%H%M%S)"
else
  echo '{}' > "${DAEMON_JSON}"
fi

python3 - <<'PY'
import json
from pathlib import Path

path = Path("/etc/docker/daemon.json")
data = json.loads(path.read_text() or "{}")
data["data-root"] = "/mnt/E/docker"
path.write_text(json.dumps(data, indent=4) + "\n")
print("    data-root =", data["data-root"])
print(path.read_text())
PY

echo "==> [5/7] Docker 재시작"
systemctl start containerd
systemctl start docker
sleep 2
systemctl is-active docker

echo "==> [6/7] 검증"
docker info 2>/dev/null | grep -E 'Docker Root Dir|Storage Driver|Server Version' || true
docker system df || true
ROOT_DIR="$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || true)"
if [[ "${ROOT_DIR}" != "${DOCKER_NEW}" ]]; then
  echo "ERROR: Docker Root Dir 이 예상과 다릅니다: '${ROOT_DIR}' (expected ${DOCKER_NEW})"
  echo "       이전 데이터는 아직 삭제하지 않았습니다."
  exit 1
fi

echo "==> [7/7] 기존 ${DOCKER_OLD} 백업 후 제거 (루트 공간 확보)"
if [[ -d "${DOCKER_OLD}" ]]; then
  BACKUP="${DOCKER_OLD}.moved-$(date +%Y%m%d%H%M%S)"
  mv "${DOCKER_OLD}" "${BACKUP}"
  echo "    이동됨: ${BACKUP}"
  echo "    삭제 중..."
  rm -rf "${BACKUP}"
  echo "    삭제 완료"
fi
# data-root 가 바뀌었으므로 빈 디렉터리는 남겨둘 필요 없음.
# 혹시 패키지가 기대하면 빈 디렉터리 생성
mkdir -p "${DOCKER_OLD}"

echo
echo "========================================"
echo "완료. Docker data-root = ${DOCKER_NEW}"
df -h / "${E_MNT}"
echo "========================================"
echo "참고: 예전에 자동 마운트되던 ${AUTO_MOUNT} 대신 ${E_MNT} 를 사용하세요."
echo "      (openpi, droid_* 등은 ${E_MNT}/ 아래에 그대로 있습니다.)"
