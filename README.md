# powersgd-adaptive-rank

## 日本語版
PowerSGD の適応ランク制御を用いた分散学習実験用リポジトリです。

### 1. このリポジトリで実施すること
このリポジトリでは、適応的な PowerSGD ランク制御を使った分散学習実験を実行します。
標準の実行構成は **2 台構成**です。

- `master` ノード: rank 0 プロセス
- `worker` ノード: rank 1 プロセス

主な実行エントリ:

- 学習エントリ: `src/main.py`
- 学習ループと適応ロジック: `src/train/train.py`
- バッチ実験ランナー: `run_all_experiments.sh`
- detached での macvlan 起動: `macvlan_run/macvlan_run.sh`

設定ファイル例は `configs/yyyymmdd/` にあります（テンプレート用途）。

### 2. ディレクトリ概要（主要部分のみ）
- `src/`: 学習・評価・コアロジック
- `configs/yyyymmdd/`: 実験設定の例
- `run_all_experiments.sh`: 設定ファイル一覧を順次実行
- `macvlan_run/macvlan_run.sh`: macvlan で detached コンテナを起動
- `tests/test_rank_logic.py`: ランク決定ロジック向け pytest
- `src/visualize/`: 結果可視化ノートブック
- `.venv_viz/`: 可視化用 Python 仮想環境

### 3. 前提条件
- 同一 L2/L3 ネットワークで到達可能な 2 台のマシン（またはホスト）
- NVIDIA driver + Docker + NVIDIA Container Toolkit
- 各ノード 1 GPU（このリポジトリでは各ノードで `--local_rank 0` を想定）

### 4. 実験手順（2 台構成）
#### Step 0: ネットワーク/コンテナ用パラメータを決める
固定 IP をそのまま流用せず、実行環境に合わせて決めてください。

- `PARENT_IF`: 外部通信に使う物理 NIC 名（例: `eth0`, `enp...`）
- `SUBNET`: macvlan 用サブネット（利用環境で疎通可能な CIDR）
- `GATEWAY`: 上記サブネットのゲートウェイ IP
- `CONTAINER_IP`:
  - ノードごとに一意
  - `SUBNET` 内のアドレス
  - 他ホスト/他コンテナで未使用
- `NAME`: コンテナ名（`dl-node-master`, `dl-node-worker` など）

master/worker で `CONTAINER_IP` と `NAME` は必ず変えてください。

#### Step 1: Docker イメージを作成（両ノードで実行）
```bash
docker build -t pytorch-cuda11.7 . --no-cache
```

#### Step 2: macvlan + detached でコンテナ起動（両ノードで実行）
master 側（プレースホルダは置換）:
```bash
PARENT_IF=<parent_interface> \
SUBNET=<subnet_cidr> \
GATEWAY=<gateway_ip> \
CONTAINER_IP=<master_container_ip> \
NAME=dl-node-master \
./macvlan_run/macvlan_run.sh
```

worker 側:
```bash
PARENT_IF=<parent_interface> \
SUBNET=<subnet_cidr> \
GATEWAY=<gateway_ip> \
CONTAINER_IP=<worker_container_ip> \
NAME=dl-node-worker \
./macvlan_run/macvlan_run.sh
```

#### Step 3: 先に worker の学習プロセスを起動
```bash
docker exec -it dl-node-worker bash
cd /workspace
chmod +x run_all_experiments.sh

MASTER_ADDR=<master_container_ip> \
WORLD_SIZE=2 \
NCCL_IFNAME=<nccl_interface> \
nohup ./run_all_experiments.sh 1 > run_rank1.log 2>&1 &
exit
```

#### Step 4: master の学習プロセスを起動
```bash
docker exec -it dl-node-master bash
cd /workspace
chmod +x run_all_experiments.sh

MASTER_ADDR=<master_container_ip> \
WORLD_SIZE=2 \
NCCL_IFNAME=<nccl_interface> \
nohup ./run_all_experiments.sh 0 > run_rank0.log 2>&1 &
exit
```

#### Step 5: 進行状況を確認
```bash
# master コンテナ
docker exec -it dl-node-master bash -lc "tail -f /workspace/run_rank0.log"

# worker コンテナ
docker exec -it dl-node-worker bash -lc "tail -f /workspace/run_rank1.log"
```

### 5. Config 設定ガイド (`configs/yyyymmdd/*.yaml`)
1 ファイルが 1 実験を表します。主なセクション:

- `save_dir`, `benchmark_save_dir`:
  - 結果保存先
  - 典型例: `/workspace/results/<date>/<experiment_name>/...`
- `seed`: 乱数シード
- `model`: モデル種別とクラス数
- `dataset`: データセット種別とバッチサイズ
- `training`: epoch 数、学習率、scheduler、warmup
- `powerSGD`: PowerSGD フック関連設定
- `compression`:
  - `enable_powerSGD`
  - `start_power_iterations`
  - ランク境界 (`min_rank`, `max_rank`, `base_rank`)
  - `calibration` 設定（候補ランク、閾値、計測イテレーション）
- `accordion`: 適応ランク切替ロジック設定
- `network_limits`: 帯域/遅延制御の設定

推奨フロー:
1. `configs/yyyymmdd/` の example をコピー
2. 実験名に合わせてリネーム
3. `save_dir` と `benchmark_save_dir` を更新
4. `compression.calibration` と `accordion` を調整
5. `run_all_experiments.sh` の `EXPERIMENT_CONFIGS` に追加

### 6. バッチランナーの挙動 (`run_all_experiments.sh`)
- 第 1 引数 `0`: master rank、`1`: worker rank
- `EXPERIMENT_CONFIGS` の設定を順番に実行
- 実験ごとのポートは `MASTER_PORT_BASE + index` で決定
- 第 2 引数 `start_from` は省略可:
  - 数値 index、または
  - config ファイル名

例:
```bash
# 先頭から実行（master 側）
./run_all_experiments.sh 0

# index 1 から再開（worker 側）
./run_all_experiments.sh 1 1
```

### 7. 最小検証
コンテナ内でロジックテストを実行:
```bash
python3 -m pytest -q tests
```

### 8. 可視化（Notebook）
可視化ノートブックは `src/visualize/` にあります。

- `viz_seed_average_main.ipynb`: シード平均・比較用の主要ノートブック

`viz_seed_average_main.ipynb` で実施する主な可視化:

- Summary table:
  - 各手法の最終精度、所要時間、平均速度（iterations/sec）を比較
  - シード平均と分散（エラーバー）を確認
- Rank changes over time:
  - adaptive 手法の rank 推移を時系列で可視化
  - 判定に使う勾配変化率を同時に確認
- Time breakdown:
  - 1 iteration あたりの compute / communication / compression / decompression を比較

#### 8.1 可視化用仮想環境を使う
このリポジトリには可視化用途の仮想環境 `.venv_viz/` を同梱しています。

```bash
cd /workspace
source .venv_viz/bin/activate
python -V
```

#### 8.2 Notebook の起動
```bash
cd /workspace
source .venv_viz/bin/activate
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

#### 8.3 Notebook 実行時の前提
- `RESULT_ROOTS` などの入力パスを、実験結果ディレクトリに合わせて更新してください。
- 設定・結果の命名を変えた場合は、ノートブック内の `glob` パターンも合わせて更新してください。

---

## English
Bandwidth-aware adaptive rank control for PowerSGD in distributed deep learning.

## 1. What This Repository Does
This repository runs distributed training experiments with adaptive PowerSGD rank control.
The standard execution model is a **2-node setup**:

- `master` node: rank 0 process
- `worker` node: rank 1 process

Main execution entry points:

- Training entry: `src/main.py`
- Training loop and adaptive logic: `src/train/train.py`
- Batch experiment runner: `run_all_experiments.sh`
- Detached macvlan launcher: `macvlan_run/macvlan_run.sh`

Example configs are in `configs/yyyymmdd/` and are intended as templates.

## 2. Directory Overview (Essential Only)
- `src/`: Training/evaluation/core logic
- `configs/yyyymmdd/`: Example experiment configs
- `run_all_experiments.sh`: Runs listed configs sequentially
- `macvlan_run/macvlan_run.sh`: Starts detached Docker container with macvlan networking
- `tests/test_rank_logic.py`: Logic-level pytest for rank decision behavior
- `src/visualize/`: Visualization notebooks
- `.venv_viz/`: Python virtual environment for visualization

## 3. Prerequisites
- Two machines (or two hosts) reachable over the same L2/L3 network
- NVIDIA driver + Docker + NVIDIA Container Toolkit
- One GPU per node (this repo assumes `--local_rank 0` per node process)

## 4. Step-by-Step Experiment Procedure (2 Nodes)
### Step 0: Choose network/container parameters
Do **not** copy fixed IPs from examples. Decide values for your environment.

- `PARENT_IF`: Physical NIC name used for external communication (for example `eth0`, `enp...`)
- `SUBNET`: Subnet used for macvlan network (must be routable/reachable in your environment)
- `GATEWAY`: Gateway IP for that subnet
- `CONTAINER_IP`:
  - unique per node
  - inside `SUBNET`
  - not used by any host/container
- `NAME`: container name (`dl-node-master`, `dl-node-worker`, etc.)

Use different `CONTAINER_IP` and `NAME` on master/worker.

### Step 1: Build Docker image (run on both nodes)
```bash
docker build -t pytorch-cuda11.7 . --no-cache
```

### Step 2: Start detached container with macvlan (run on both nodes)
Master node example (replace placeholders):
```bash
PARENT_IF=<parent_interface> \
SUBNET=<subnet_cidr> \
GATEWAY=<gateway_ip> \
CONTAINER_IP=<master_container_ip> \
NAME=dl-node-master \
./macvlan_run/macvlan_run.sh
```

Worker node example:
```bash
PARENT_IF=<parent_interface> \
SUBNET=<subnet_cidr> \
GATEWAY=<gateway_ip> \
CONTAINER_IP=<worker_container_ip> \
NAME=dl-node-worker \
./macvlan_run/macvlan_run.sh
```

### Step 3: Launch worker training process first
```bash
docker exec -it dl-node-worker bash
cd /workspace
chmod +x run_all_experiments.sh

MASTER_ADDR=<master_container_ip> \
WORLD_SIZE=2 \
NCCL_IFNAME=<nccl_interface> \
nohup ./run_all_experiments.sh 1 > run_rank1.log 2>&1 &
exit
```

### Step 4: Launch master training process
```bash
docker exec -it dl-node-master bash
cd /workspace
chmod +x run_all_experiments.sh

MASTER_ADDR=<master_container_ip> \
WORLD_SIZE=2 \
NCCL_IFNAME=<nccl_interface> \
nohup ./run_all_experiments.sh 0 > run_rank0.log 2>&1 &
exit
```

### Step 5: Monitor progress
```bash
# on master container
docker exec -it dl-node-master bash -lc "tail -f /workspace/run_rank0.log"

# on worker container
docker exec -it dl-node-worker bash -lc "tail -f /workspace/run_rank1.log"
```

## 5. Config File Guide (`configs/yyyymmdd/*.yaml`)
Each config controls one experiment. Main sections:

- `save_dir`, `benchmark_save_dir`:
  - output directories
  - usually set to `/workspace/results/<date>/<experiment_name>/...`
- `seed`: random seed
- `model`: model type and class count
- `dataset`: dataset type and batch size
- `training`: epochs, optimizer LR, scheduler, warmup
- `powerSGD`: hook-level PowerSGD options
- `compression`:
  - `enable_powerSGD`
  - `start_power_iterations`
  - rank bounds (`min_rank`, `max_rank`, `base_rank`)
  - `calibration` options (rank candidates, thresholds, iteration counts)
- `accordion`: adaptive rank-switch behavior
- `network_limits`: bandwidth/latency shaping switches and parameters

Recommended workflow:
1. Copy one example config in `configs/yyyymmdd/`.
2. Rename it to your experiment name.
3. Update `save_dir` and `benchmark_save_dir`.
4. Adjust `compression.calibration` and `accordion` parameters.
5. Add the config path to `EXPERIMENT_CONFIGS` in `run_all_experiments.sh`.

## 6. Batch Runner Behavior (`run_all_experiments.sh`)
- Argument `0` runs master rank, `1` runs worker rank.
- Script runs all configs listed in `EXPERIMENT_CONFIGS` sequentially.
- Per-experiment port is derived from `MASTER_PORT_BASE + index`.
- Optional second argument `start_from`:
  - numeric index, or
  - config filename

Examples:
```bash
# run all from first config (master side)
./run_all_experiments.sh 0

# resume from index 1 (worker side)
./run_all_experiments.sh 1 1
```

## 7. Minimal Verification
Run logic tests inside container:
```bash
python3 -m pytest -q tests
```

## 8. Visualization (Notebook)
Visualization notebooks are stored in `src/visualize/`.

- `viz_seed_average_main.ipynb`: Main notebook for seed-averaged comparison

Main visualizations included in `viz_seed_average_main.ipynb`:

- Summary table:
  - Compare final accuracy, elapsed time, and average throughput (iterations/sec)
  - Show seed-averaged values with variation (error bars)
- Rank changes over time:
  - Plot adaptive rank transitions over epochs
  - Overlay gradient change rate used for rank decision
- Time breakdown:
  - Compare per-iteration compute / communication / compression / decompression time

### 8.1 Use the visualization virtual environment
This repository includes `.venv_viz/` for notebook-based analysis.

```bash
cd /workspace
source .venv_viz/bin/activate
python -V
```

### 8.2 Launch Jupyter
```bash
cd /workspace
source .venv_viz/bin/activate
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### 8.3 Notebook assumptions
- Update input paths such as `RESULT_ROOTS` to your actual results directory.
- If you change config/result naming rules, update notebook `glob` patterns accordingly.
