#!/usr/bin/env bash
set -e

# ---- 必要ならここだけ変える（元スクリプトと同じ）----
PARENT_IF=${PARENT_IF:-eth0}
SUBNET=${SUBNET:-192.168.10.0/24}
GATEWAY=${GATEWAY:-192.168.10.1}
NET_NAME=${NET_NAME:-macvlan_net}
CONTAINER_IP=${CONTAINER_IP:-192.168.10.20}

IMAGE=${IMAGE:-pytorch-cuda11.7}
NAME=${NAME:-dl-node}
MOUNT=${MOUNT:-$(pwd):/workspace}
WORKDIR=${WORKDIR:-/workspace}
GPUS=${GPUS:-all}
# ---------------------------------

# macvlanネットワーク作成（あればスキップ）
if ! docker network inspect "$NET_NAME" >/dev/null 2>&1; then
  docker network create -d macvlan \
    --subnet "$SUBNET" --gateway "$GATEWAY" \
    -o parent="$PARENT_IF" "$NET_NAME"
fi

# コンテナを「バックグラウンド常駐」で起動
# PID1 は sleep infinity にしておき、必要なときは docker exec で入る
docker run -d --rm \
  --name "$NAME" \
  --network "$NET_NAME" --ip "$CONTAINER_IP" \
  --cap-add NET_ADMIN \
  --gpus "$GPUS" \
  -v "$MOUNT" \
  -w "$WORKDIR" \
  "$IMAGE" sleep infinity

echo "Started container: $NAME ($CONTAINER_IP) on network $NET_NAME"
