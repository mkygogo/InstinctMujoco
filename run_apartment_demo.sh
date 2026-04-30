#!/bin/bash
# 两室一厅公寓导航 demo — 启动全部服务
# 用法: bash run_apartment_demo.sh

set -e

# 杀掉旧进程
fuser -k 9876/tcp 2>/dev/null || true
fuser -k 8190/tcp 2>/dev/null || true
fuser -k 8765/tcp 2>/dev/null || true
sleep 0.5

echo "=== [1/4] 启动场景中继 (ws://0.0.0.0:8190) ==="
cd ~/StereoSpatial/SpatialCanvas
node scene-relay.mjs &
RELAY_PID=$!
sleep 1

echo "=== [2/4] 启动 MuJoCo Frame Server (apartment, port 9876) ==="
cd ~/ProjectInstinct/mujoco/InstinctMujoco
source ~/ProjectInstinct/mujoco/InstinctMJ/.venv/bin/activate
DISPLAY=:0 python run_mujoco_frame_server.py --port 9876 --terrain apartment &
MUJOCO_PID=$!
sleep 3

echo "=== [3/4] 启动感知服务 (FFS, port 8765) ==="
cd ~/StereoSpatial/Fast-FoundationStereo
conda run -n ffs --no-banner python scripts/run_perception_service.py \
  --camera mujoco --mujoco-host 127.0.0.1 --mujoco-port 9876 \
  --detect-source left_ir \
  --transport websocket --host 0.0.0.0 --port 8765 \
  --relay-url ws://127.0.0.1:8190 \
  --scene-refresh-interval 5 --scene-max-depth 12 &
FFS_PID=$!

echo "=== [4/4] 全部启动完成 ==="
echo "  Frame Server PID: $MUJOCO_PID"
echo "  Scene Relay  PID: $RELAY_PID"
echo "  Perception   PID: $FFS_PID"
echo ""
echo "打开浏览器: http://localhost:5173/#robot-control"
echo "按 Ctrl+C 停止所有服务"

# 等待任一进程退出或 Ctrl+C
trap "echo '停止所有服务...'; kill $MUJOCO_PID $RELAY_PID $FFS_PID 2>/dev/null; exit 0" INT TERM
wait
