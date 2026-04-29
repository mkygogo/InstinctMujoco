# InstinctMujoco

基于纯 MuJoCo 的 Unitree G1 人形机器人 Parkour 策略部署框架。将 InstinctMJ 训练导出的 ONNX 策略在独立 MuJoCo 环境中运行，无需 Isaac Lab / GPU 仿真器。

## 项目简介

本项目实现了 G1 29-DOF 人形机器人在 MuJoCo 物理引擎中的 parkour 策略推理，核心功能包括：

- **ONNX 策略推理** — 加载 depth encoder + actor 网络，支持 CPU/CUDA
- **光线投射深度感知** — 使用 `mj_multiRay` 实现训练一致的 `distance_to_image_plane` 深度图，包含时序堆叠（37帧缓冲区，skip=5，输出8帧）
- **BeyondMimic 执行器** — 与训练环境完全匹配的 PD 位置控制器参数（刚度、阻尼、力矩限制、电枢惯量）
- **场景构建** — 自动生成地面、天空盒、光照、楼梯等地形
- **航向校正** — 内置 P 控制器补偿偏航漂移，保持机器人直线行走

## 项目结构

```
InstinctMujoco/
├── run_parkour_mujoco.py   # 主运行入口，MuJoCo 仿真循环
├── run_goto_demo.py        # 交互式导航 Demo（双击选点，机器人自动走过去）
├── parkour_onnx_policy.py  # ONNX 策略封装与观测历史管理
├── robot_config.py         # G1 机器人配置（执行器参数、关节位姿、相机偏移）
├── scene_builder.py        # MuJoCo 场景构建（物理参数、地形、执行器）
├── math_utils.py           # 四元数运算、深度图处理工具
├── run_demo.sh             # 一键启动脚本
├── models/
│   └── parkour/            # 预训练 ONNX 模型权重
│       ├── 0-depth_encoder.onnx
│       ├── actor.onnx
│       ├── actor.onnx.data
│       ├── metadata.json
│       └── policy_normalizer.npz
└── mjcf/
    ├── g1_29dof_torsoBase_popsicle_with_shoe.xml  # G1 机器人 MJCF 模型
    └── meshes/             # STL 网格文件
```

## 环境要求

- Python ≥ 3.10
- MuJoCo ≥ 3.0（推荐 3.5+）
- 有显示器环境（带 viewer）或无头模式（EGL）

## 安装

### 1. 克隆仓库

```bash
git clone git@github.com:mkygogo/InstinctMujoco.git
cd InstinctMujoco
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install mujoco numpy onnxruntime opencv-python
```

本项目还依赖 [mjlab](https://github.com/ProjectInstinct/mjlab)（用于执行器配置和场景工具）。请按 mjlab 的说明安装：

```bash
# 假设 mjlab 在上级目录
pip install -e ../mjlab
```

## 快速开始

### 一键运行楼梯攀爬 Demo

```bash
./run_demo.sh
```

这将启动带有 viewer 的仿真，机器人以 0.5 m/s 速度攀爬楼梯。

### 交互式导航 Demo（双击选点）

```bash
python run_goto_demo.py
```

启动后机器人原地站立。在场景中 **双击地面** 即可设置目标点，机器人会自动行走到该位置并停下，等待下一个目标。

操作方式：
- **双击**：在地面选取导航目标
- **Backspace**：取消当前目标
- **Esc**：退出
- **左键拖动**：旋转视角 / **右键拖动**：平移视角 / **滚轮**：缩放

### 自定义运行

```bash
# 平地行走
python run_parkour_mujoco.py --command-x 0.5

# 楼梯攀爬（带深度感知）
python run_parkour_mujoco.py --command-x 0.5 --terrain stairs --use-depth

# 无头模式运行（无显示器环境）
python run_parkour_mujoco.py --command-x 0.5 --terrain stairs --use-depth --headless --no-real-time

# 调试模式（每步打印状态）
python run_parkour_mujoco.py --command-x 0.5 --terrain stairs --use-depth --log-interval 1
```

### 全部命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--command-x` | 0.0 | 前进速度指令 (m/s) |
| `--command-y` | 0.0 | 侧向速度指令 (m/s) |
| `--command-yaw` | 0.0 | 旋转速度指令 (rad/s) |
| `--terrain` | flat | 地形类型：`flat` 或 `stairs` |
| `--use-depth` | 关闭 | 启用光线投射深度感知 |
| `--steps` | 5000 | 仿真步数 |
| `--yaw-correction-gain` | 0.5 | 航向校正 P 增益（0 禁用） |
| `--headless` | 关闭 | 无头模式（不启动 viewer） |
| `--no-real-time` | 关闭 | 不限制实时速度 |
| `--action-clip` | 100.0 | 动作裁剪范围 |
| `--target-smoothing` | 1.0 | 目标位置平滑系数 |
| `--log-interval` | 0 | 日志打印间隔（0 不打印） |
| `--stand-only` | 关闭 | 仅站立（不执行策略） |

## 技术细节

### 深度感知流水线

训练环境使用光线投射（raycast）而非 OpenGL 渲染器获取深度图。本项目完整复现了训练流水线：

1. **针孔相机光线生成** — 64×36 分辨率，fovy=58.29°
2. **偏航对齐** — 相机姿态仅保留偏航角，消除俯仰和翻滚
3. **`mj_multiRay` 光线投射** — 使用 `geomgroup=[1,0,0,0,0,0]` 仅检测地形，排除机器人自身
4. **距离到像平面投影** — `depth = dist × ray_direction_x`
5. **后处理** — 裁剪 (18,0,16,16) → 双线性插值至 18×32 → 高斯模糊 (k=3, σ=1) → 归一化至 [0, 2.5]
6. **时序堆叠** — 37帧原始缓冲区，每5帧采样一次，输出8帧深度栈

### 执行器模型

使用 BeyondMimic 配置的 PD 位置控制器：

$$\tau = K_p \cdot (q_{target} - q) - K_d \cdot \dot{q}$$

执行器参数从电机电枢惯量计算：$K_p = I \cdot \omega_n^2$，$K_d = 2\zeta I \omega_n$，其中 $\omega_n = 20\pi$ rad/s，$\zeta = 2.0$。

### 物理参数

| 参数 | 值 |
|------|-----|
| 仿真时间步 | 0.005s |
| 控制频率 | 50 Hz (decimation=4) |
| 积分器 | implicitfast |
| 求解器迭代 | 10 次 |
| 线搜索迭代 | 20 次 |
| CCD 迭代 | 128 次 |

## 致谢

- [MuJoCo](https://mujoco.org/) — DeepMind 物理引擎
- [InstinctMJ](https://github.com/ProjectInstinct/InstinctMJ) — 训练框架
- [mjlab](https://github.com/ProjectInstinct/mjlab) — MuJoCo 工具库
- [Unitree G1](https://www.unitree.com/g1/) — 人形机器人平台
