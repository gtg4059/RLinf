#!/usr/bin/env bash
# Migrate containerd root from /var/lib/containerd to /mnt/E/containerd
# (Docker overlayfs/io.containerd.snapshotter stores image layers here.)
# Usage: sudo bash toolkits/migrate_containerd_to_E.sh
set -euo pipefail

E_MNT="/mnt/E"
CTD_NEW="${E_MNT}/containerd"
CTD_OLD="/var/lib/containerd"
CTD_CFG="/etc/containerd/config.toml"

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: root 권한이 필요합니다. 다음으로 실행하세요:"
  echo "  sudo bash $0"
  exit 1
fi

if ! mountpoint -q "${E_MNT}"; then
  echo "ERROR: ${E_MNT} 가 마운트되어 있지 않습니다. 먼저 migrate_docker_to_E.sh 를 실행하세요."
  exit 1
fi

echo "==> [1/6] 현재 사용량"
df -h / "${E_MNT}"
du -sh "${CTD_OLD}" || true

echo "==> [2/6] Docker / containerd 중지"
systemctl stop docker.socket docker containerd || true
sleep 2
if pgrep -x dockerd >/dev/null 2>&1 || pgrep -x containerd >/dev/null 2>&1; then
  echo "ERROR: dockerd/containerd 가 아직 실행 중입니다."
  pgrep -a -x 'dockerd|containerd' || true
  exit 1
fi

echo "==> [3/6] 데이터 복사: ${CTD_OLD} → ${CTD_NEW}"
mkdir -p "${CTD_NEW}"
rsync -aHAX --info=progress2 "${CTD_OLD}/" "${CTD_NEW}/"

echo "==> [4/6] containerd root 설정: ${CTD_CFG}"
cp -a "${CTD_CFG}" "${CTD_CFG}.bak.$(date +%Y%m%d%H%M%S)"

python3 - <<'PY'
from pathlib import Path

path = Path("/etc/containerd/config.toml")
text = path.read_text()
lines = text.splitlines()
out = []
replaced_root = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith("root ") or stripped.startswith("#root ") or stripped.startswith("root="):
        out.append('root = "/mnt/E/containerd"')
        replaced_root = True
    else:
        out.append(line)
if not replaced_root:
    # insert near top after comments/disabled_plugins
    insert_at = 0
    for i, line in enumerate(out):
        if line.strip() and not line.strip().startswith("#"):
            insert_at = i + 1
            break
    out.insert(insert_at, 'root = "/mnt/E/containerd"')
path.write_text("\n".join(out) + "\n")
print(path.read_text())
PY

echo "==> [5/6] 서비스 재시작 및 검증"
systemctl start containerd
systemctl start docker
sleep 3
systemctl is-active containerd docker

# containerd root 확인
if command -v containerd >/dev/null; then
  containerd config dump 2>/dev/null | grep -E '^root\s*=' || true
fi

docker info 2>/dev/null | grep -E 'Docker Root Dir|Storage Driver|Server Version' || true
docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}' || true

# 새 경로에 데이터가 있는지 확인
NEW_SIZE="$(du -sb "${CTD_NEW}" | awk '{print $1}')"
if [[ "${NEW_SIZE}" -lt 1000000000 ]]; then
  echo "ERROR: ${CTD_NEW} 크기가 비정상적으로 작습니다 (${NEW_SIZE} bytes). 기존 데이터는 삭제하지 않습니다."
  exit 1
fi

# 이미지가 실제로 새 store에서 동작하는지 스모크 테스트
docker run --rm alpine echo "containerd_migrate_ok"

echo "==> [6/6] 기존 ${CTD_OLD} 제거 (루트 공간 확보)"
BACKUP="${CTD_OLD}.moved-$(date +%Y%m%d%H%M%S)"
mv "${CTD_OLD}" "${BACKUP}"
echo "    이동됨: ${BACKUP}"
echo "    삭제 중... (수 분 걸릴 수 있음)"
rm -rf "${BACKUP}"
mkdir -p "${CTD_OLD}"

echo
echo "========================================"
echo "완료. containerd root = ${CTD_NEW}"
df -h / "${E_MNT}"
du -sh "${CTD_NEW}"
echo "========================================"
