import time
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Any, List, Optional, Tuple
import psutil
import os
from dataclasses import dataclass
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import (
    powerSGD_hook,
    PowerSGDState
)
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


@dataclass
class PowerSGDBenchmarkResult:
    """PowerSGD計測結果を格納するデータクラス"""
    epoch: int
    iteration: int
    communication_time: float
    compression_time: float
    decompression_time: float
    total_processing_time: float
    iteration_total_time: float
    compute_time: float
    memory_usage_mb: float
    gpu_memory_usage_mb: float
    compression_ratio: float
    bandwidth_mbps: float
    loss: float
    accuracy: Optional[float] = None
    powerSGD_active: bool = False
    current_rank: int = 0


class PowerSGDBenchmark:
    """PowerSGDの計算速度を計測するクラス"""
    
    def __init__(self, rank: int, world_size: int, save_dir: str = "./benchmark_results"):
        self.rank = rank
        self.world_size = world_size
        self.base_save_dir = save_dir
        
        # タイムスタンプ付きのサブディレクトリを作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f"benchmark_{timestamp}")
        
        # サブディレクトリ構造を作成
        self.csv_dir = os.path.join(self.save_dir, "csv")
        self.visualizations_dir = os.path.join(self.save_dir, "visualizations")
        self.configs_dir = os.path.join(self.save_dir, "configs")
        
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)
        
        self.results: List[PowerSGDBenchmarkResult] = []
        self.start_time = None
        self.communication_start_time = None
        self.compression_start_time = None
        self.decompression_start_time = None
        
        # 直近の計測値を保持
        self.last_compression_time = 0.0
        self.last_decompression_time = 0.0
        # Accumulate comm time per iteration (sum over all buckets)
        self.last_communication_time = 0.0

        # epochごとのaccuracyを記録
        self.epoch_acc: List[Tuple[int, int, float]] = []  # (epoch, iteration_idx, accuracy_%)

        
    def start_iteration(self):
        """イテレーション開始時の計測開始"""
        self.start_time = time.perf_counter()
        
    def start_communication(self):
        """通信開始時の計測開始"""
        self.communication_start_time = time.perf_counter()
        
    def end_communication(self) -> float:
        """通信終了時の計測終了"""
        if self.communication_start_time is None:
            return 0.0
        communication_time = time.perf_counter() - self.communication_start_time
        self.communication_start_time = None
        return communication_time
        
    def start_compression(self):
        """圧縮開始時の計測開始"""
        self.compression_start_time = time.perf_counter()
        
    def end_compression(self) -> float:
        """圧縮終了時の計測終了"""
        if self.compression_start_time is None:
            return 0.0
        compression_time = time.perf_counter() - self.compression_start_time
        self.compression_start_time = None
        self.last_compression_time = compression_time
        return compression_time
        
    def start_decompression(self):
        """復元開始時の計測開始"""
        self.decompression_start_time = time.perf_counter()
        
    def end_decompression(self) -> float:
        """復元終了時の計測終了"""
        if self.decompression_start_time is None:
            return 0.0
        decompression_time = time.perf_counter() - self.decompression_start_time
        self.decompression_start_time = None
        self.last_decompression_time = decompression_time
        return decompression_time
        
    def get_memory_usage(self) -> Tuple[float, float]:
        """メモリ使用量を取得"""
        # CPUメモリ使用量
        process = psutil.Process(os.getpid())
        cpu_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # GPUメモリ使用量
        gpu_memory_mb = 0.0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
        return cpu_memory_mb, gpu_memory_mb
        
    def calculate_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """圧縮率を計算"""
        if original_size == 0:
            return 0.0
        return (1.0 - compressed_size / original_size) * 100.0
        
    def calculate_bandwidth(self, data_size_bytes: int, communication_time: float) -> float:
        """通信帯域幅を計算（MB/s）"""
        if communication_time <= 0:
            return 0.0
        data_size_mb = data_size_bytes / (1024 * 1024)  # バイト → MB
        return data_size_mb / communication_time  # MB/s

    def record_val_accuracy(self, epoch: int, accuracy_pct: float, iteration: Optional[int] = None) -> None:
        """
        エポック終了時の Validation Accuracy(%) を記録する。
        iteration が None の場合は、その epoch の最後のイテレーション番号を自動推定。
        """
        # %表記に統一（0-1で渡されたら*100）
        if accuracy_pct <= 1.0:
            accuracy_pct *= 100.0

        if iteration is None:
            # その epoch で記録された中で最大の iteration を使う
            iters = [r.iteration for r in self.results if r.epoch == epoch]
            iteration = max(iters) if iters else (self.results[-1].iteration if self.results else 0)

        self.epoch_acc.append((epoch, int(iteration), float(accuracy_pct)))
        
    def record_result(
        self,
        epoch: int,
        iteration: int,
        communication_time: float,
        compression_time: float,
        decompression_time: float,
        original_size: int,
        compressed_size: int,
        loss: float,
        accuracy: Optional[float] = None,
        powerSGD_active: bool = False,
        current_rank: int = 0
    ):
        """計測結果を記録"""
        if self.start_time is None:
            return
            
        # 総処理時間を正しく計算（個別時間の合計）
        total_processing_time = communication_time + compression_time + decompression_time
        iteration_total_time = 0.0
        if self.start_time is not None:
            iteration_total_time = time.perf_counter() - self.start_time
        compute_time = max(0.0, iteration_total_time - total_processing_time)
        cpu_memory_mb, gpu_memory_usage_mb = self.get_memory_usage()
        compression_ratio = self.calculate_compression_ratio(original_size, compressed_size)
        
        # PowerSGD圧縮を考慮した通信帯域幅計算
        if powerSGD_active and current_rank > 0:
            # 圧縮後のデータサイズを使用（実際に通信されるサイズ）
            actual_communication_size = compressed_size
        else:
            # 圧縮なしの場合は元のサイズを使用
            actual_communication_size = original_size
        
        bandwidth_mbps = self.calculate_bandwidth(actual_communication_size, communication_time)
        
        result = PowerSGDBenchmarkResult(
            epoch=epoch,
            iteration=iteration,
            communication_time=communication_time,
            compression_time=compression_time,
            decompression_time=decompression_time,
            total_processing_time=total_processing_time,
            iteration_total_time=iteration_total_time,
            compute_time=compute_time,
            memory_usage_mb=cpu_memory_mb,
            gpu_memory_usage_mb=gpu_memory_usage_mb,
            compression_ratio=compression_ratio,
            bandwidth_mbps=bandwidth_mbps,
            loss=loss,
            accuracy=accuracy,
            powerSGD_active=powerSGD_active,
            current_rank=current_rank
        )
        
        self.results.append(result)
        
    def save_results(self, filename: str = None):
        """結果をCSVファイルに保存"""
        if not self.results:
            return
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"powerSGD_benchmark_{timestamp}.csv"
            
        filepath = os.path.join(self.csv_dir, filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [
                'epoch', 'iteration', 'communication_time', 'compression_time',
                'decompression_time', 'total_processing_time', 'iteration_total_time',
                'compute_time', 'memory_usage_mb',
                'gpu_memory_usage_mb', 'compression_ratio', 'bandwidth_mbps',
                'loss', 'accuracy', 'powerSGD_active', 'current_rank'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                writer.writerow({
                    'epoch': result.epoch,
                    'iteration': result.iteration,
                    'communication_time': result.communication_time,
                    'compression_time': result.compression_time,
                    'decompression_time': result.decompression_time,
                    'total_processing_time': result.total_processing_time,
                    'iteration_total_time': result.iteration_total_time,
                    'compute_time': result.compute_time,
                    'memory_usage_mb': result.memory_usage_mb,
                    'gpu_memory_usage_mb': result.gpu_memory_usage_mb,
                    'compression_ratio': result.compression_ratio,
                    'bandwidth_mbps': result.bandwidth_mbps,
                    'loss': result.loss,
                    'accuracy': result.accuracy,
                    'powerSGD_active': result.powerSGD_active,
                    'current_rank': result.current_rank
                })
                
        if self.rank == 0:
            print(f"Benchmark results saved to: {filepath}")

        if self.epoch_acc:
            acc_path = os.path.join(self.csv_dir, "epoch_accuracy.csv")
            with open(acc_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "iteration", "val_accuracy_pct"])
                for e, it, acc in self.epoch_acc:
                    w.writerow([e, it, acc])
            if self.rank == 0:
                print(f"Epoch accuracy saved to: {acc_path}")
            
    def create_visualizations(self, output_dir: str = None) -> None:
        """ベンチマーク結果の可視化を作成"""
        if not self.results:
            return
            
        if output_dir is None:
            output_dir = self.visualizations_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # データフレーム形式に変換
        epochs = [r.epoch for r in self.results]
        iterations = [r.iteration for r in self.results]
        comm_times = [r.communication_time for r in self.results]
        comp_times = [r.compression_time for r in self.results]
        decomp_times = [r.decompression_time for r in self.results]
        total_times = [r.total_processing_time for r in self.results]
        memory_usage = [r.memory_usage_mb for r in self.results]
        gpu_memory_usage = [r.gpu_memory_usage_mb for r in self.results]
        compression_ratios = [r.compression_ratio for r in self.results]
        bandwidths = [r.bandwidth_mbps for r in self.results]
        losses = [r.loss for r in self.results]
        accuracies = [r.accuracy if r.accuracy is not None else 0 for r in self.results]
        
        # 1. 時間関連のグラフ
        self._create_time_plots(epochs, iterations, comm_times, comp_times, decomp_times, total_times, output_dir)
        
        # 2. メモリ使用量のグラフ
        self._create_memory_plots(epochs, iterations, memory_usage, gpu_memory_usage, output_dir)
        
        # 3. 圧縮効率のグラフ
        self._create_compression_plots(epochs, iterations, compression_ratios, bandwidths, output_dir)
        
        # 4. 学習性能のグラフ
        self._create_learning_plots(epochs, iterations, losses, accuracies, output_dir)
        
        # 5. PowerSGD状態の可視化
        powerSGD_active_list = [r.powerSGD_active for r in self.results]
        current_ranks = [r.current_rank for r in self.results]
        self._create_powerSGD_status_plots(epochs, iterations, powerSGD_active_list, current_ranks, output_dir)
        
        # 6. 総合的なダッシュボード
        self._create_dashboard(epochs, iterations, comm_times, comp_times, decomp_times, 
                             memory_usage, gpu_memory_usage, compression_ratios, bandwidths, 
                             losses, accuracies, output_dir)
        
        if self.rank == 0:
            print(f"Visualizations saved to: {output_dir}")
            
    def _create_time_plots(self, epochs, iterations, comm_times, comp_times, decomp_times, total_times, output_dir):
        """時間関連のグラフを作成"""
        import numpy as np
        
        # 累積時間の計算
        comm_cumulative = np.cumsum(comm_times)
        comp_cumulative = np.cumsum(comp_times)
        decomp_cumulative = np.cumsum(decomp_times)
        total_cumulative = np.cumsum(total_times)
        
        # 平均値の計算
        comm_mean = np.mean(comm_times)
        comp_mean = np.mean(comp_times)
        decomp_mean = np.mean(decomp_times)
        total_mean = np.mean(total_times)
        
        # 1. Communication Time - 個別グラフ + 累積時間
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('PowerSGD Benchmark - Communication Time', fontsize=16)
        
        # 左側：個別時間
        ax1.plot(iterations, comm_times, 'b-', alpha=0.7, linewidth=2, label='Communication Time')
        ax1.axhline(y=comm_mean, color='red', linestyle='--', alpha=0.8, 
                   label=f'Mean: {comm_mean:.6f}s')
        ax1.set_title('Individual Communication Time')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Time (s)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 右側：累積時間
        ax2.plot(iterations, comm_cumulative, 'b-', alpha=0.7, linewidth=2, label='Cumulative Communication Time')
        ax2.set_title('Cumulative Communication Time')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Cumulative Time (s)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'communication_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Compression Time - 個別グラフ + 累積時間
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('PowerSGD Benchmark - Compression Time', fontsize=16)
        
        # 左側：個別時間
        ax1.plot(iterations, comp_times, 'g-', alpha=0.7, linewidth=2, label='Compression Time')
        ax1.axhline(y=comp_mean, color='red', linestyle='--', alpha=0.8, 
                   label=f'Mean: {comp_mean:.6f}s')
        ax1.set_title('Individual Compression Time')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Time (s)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 右側：累積時間
        ax2.plot(iterations, comp_cumulative, 'g-', alpha=0.7, linewidth=2, label='Cumulative Compression Time')
        ax2.set_title('Cumulative Compression Time')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Cumulative Time (s)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'compression_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Decompression Time - 個別グラフ + 累積時間
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('PowerSGD Benchmark - Decompression Time', fontsize=16)
        
        # 左側：個別時間
        ax1.plot(iterations, decomp_times, 'r-', alpha=0.7, linewidth=2, label='Decompression Time')
        ax1.axhline(y=decomp_mean, color='red', linestyle='--', alpha=0.8, 
                   label=f'Mean: {decomp_mean:.6f}s')
        ax1.set_title('Individual Decompression Time')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Time (s)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 右側：累積時間
        ax2.plot(iterations, decomp_cumulative, 'r-', alpha=0.7, linewidth=2, label='Cumulative Decompression Time')
        ax2.set_title('Cumulative Decompression Time')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Cumulative Time (s)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'decompression_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Total Processing Time - 個別グラフ + 累積時間
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('PowerSGD Benchmark - Total Processing Time', fontsize=16)
        
        # 左側：個別時間
        ax1.plot(iterations, total_times, 'purple', alpha=0.7, linewidth=2, label='Total Processing Time')
        ax1.axhline(y=total_mean, color='red', linestyle='--', alpha=0.8, 
                   label=f'Mean: {total_mean:.6f}s')
        ax1.set_title('Individual Total Processing Time')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Time (s)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 右側：累積時間
        ax2.plot(iterations, total_cumulative, 'purple', alpha=0.7, linewidth=2, label='Cumulative Total Processing Time')
        ax2.set_title('Cumulative Total Processing Time')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Cumulative Time (s)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'total_processing_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 9. 個別時間の比較グラフ
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PowerSGD Benchmark - Individual Time Metrics', fontsize=16)
        
        axes[0, 0].plot(iterations, comm_times, 'b-', alpha=0.7, linewidth=2, label='Communication Time')
        axes[0, 0].axhline(y=comm_mean, color='red', linestyle='--', alpha=0.8, 
                           label=f'Mean: {comm_mean:.6f}s')
        axes[0, 0].set_title('Communication Time')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Time (s)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].plot(iterations, comp_times, 'g-', alpha=0.7, linewidth=2, label='Compression Time')
        axes[0, 1].axhline(y=comp_mean, color='red', linestyle='--', alpha=0.8, 
                           label=f'Mean: {comp_mean:.6f}s')
        axes[0, 1].set_title('Compression Time')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Time (s)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        axes[1, 0].plot(iterations, decomp_times, 'r-', alpha=0.7, linewidth=2, label='Decompression Time')
        axes[1, 0].axhline(y=decomp_mean, color='red', linestyle='--', alpha=0.8, 
                           label=f'Mean: {decomp_mean:.6f}s')
        axes[1, 0].set_title('Decompression Time')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Time (s)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].plot(iterations, total_times, 'purple', alpha=0.7, linewidth=2, label='Total Processing Time')
        axes[1, 1].axhline(y=total_mean, color='red', linestyle='--', alpha=0.8, 
                           label=f'Mean: {total_mean:.6f}s')
        axes[1, 0].set_title('Total Processing Time')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'individual_time_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 10. 累積時間の比較グラフ
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PowerSGD Benchmark - Cumulative Time Metrics', fontsize=16)
        
        axes[0, 0].plot(iterations, comm_cumulative, 'b-', alpha=0.7, linewidth=2, label='Cumulative Communication Time')
        axes[0, 0].set_title('Cumulative Communication Time')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Cumulative Time (s)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].plot(iterations, comp_cumulative, 'g-', alpha=0.7, linewidth=2, label='Cumulative Compression Time')
        axes[0, 1].set_title('Cumulative Compression Time')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Cumulative Time (s)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        axes[1, 0].plot(iterations, decomp_cumulative, 'r-', alpha=0.7, linewidth=2, label='Cumulative Decompression Time')
        axes[1, 0].set_title('Cumulative Decompression Time')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Cumulative Time (s)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].plot(iterations, total_cumulative, 'purple', alpha=0.7, linewidth=2, label='Cumulative Total Processing Time')
        axes[1, 1].set_title('Cumulative Total Processing Time')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Cumulative Time (s)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cumulative_time_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 11. 統計サマリー（平均値と標準偏差）
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PowerSGD Benchmark - Time Statistics Summary', fontsize=16)
        
        # 平均値と標準偏差の計算
        comm_std = np.std(comm_times)
        comp_std = np.std(comp_times)
        decomp_std = np.std(decomp_times)
        total_std = np.std(total_times)
        
        # バーチャートで平均値を表示
        metrics = ['Communication', 'Compression', 'Decompression', 'Total']
        means = [comm_mean, comp_mean, decomp_mean, total_mean]
        stds = [comm_std, comp_std, decomp_std, total_std]
        colors = ['blue', 'green', 'red', 'purple']
        
        bars = axes[0, 0].bar(metrics, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        axes[0, 0].set_title('Average Time with Standard Deviation')
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Time (s)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 値のラベルを追加
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + std + 0.0001,
                           f'{mean:.6f}\n±{std:.6f}', ha='center', va='bottom', fontsize=10)
        
        # 累積時間の比較
        axes[0, 1].plot(iterations, comm_cumulative, 'b-', alpha=0.7, linewidth=2, label='Communication')
        axes[0, 1].plot(iterations, comp_cumulative, 'g-', alpha=0.7, linewidth=2, label='Compression')
        axes[0, 1].plot(iterations, decomp_cumulative, 'r-', alpha=0.7, linewidth=2, label='Decompression')
        axes[0, 1].plot(iterations, total_cumulative, 'purple', alpha=0.7, linewidth=2, label='Total')
        axes[0, 1].set_title('Cumulative Time Comparison')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Cumulative Time (s)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 個別時間の比較
        axes[1, 0].plot(iterations, comm_times, 'b-', alpha=0.7, linewidth=1, label='Communication')
        axes[1, 0].plot(iterations, comp_times, 'g-', alpha=0.7, linewidth=1, label='Compression')
        axes[1, 0].plot(iterations, decomp_times, 'r-', alpha=0.7, linewidth=1, label='Decompression')
        axes[1, 0].set_title('Individual Time Comparison')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Time (s)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # 統計情報のテキスト表示
        stats_text = f"""Time Statistics:
        Communication: {comm_mean:.6f}s ± {comm_std:.6f}s
        Compression: {comp_mean:.6f}s ± {comp_std:.6f}s
        Decompression: {decomp_mean:.6f}s ± {decomp_std:.6f}s
        Total: {total_mean:.6f}s ± {total_std:.6f}s"""
                
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].set_title('Statistical Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_statistics_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 12. インタラクティブ版（Plotly）
        # 個別時間メトリクス
        fig = make_subplots(rows=2, cols=2, subplot_titles=('Communication Time', 'Compression Time', 
                                                           'Decompression Time', 'Total Processing Time'))
        
        fig.add_trace(go.Scatter(x=iterations, y=comm_times, mode='lines', name='Comm Time', 
                                line=dict(color='blue'), hovertemplate='Iteration: %{x}<br>Time: %{y:.6f}s<extra></extra>'), row=1, col=1)
        fig.add_trace(go.Scatter(x=iterations, y=comp_times, mode='lines', name='Comp Time', 
                                line=dict(color='green'), hovertemplate='Iteration: %{x}<br>Time: %{y:.6f}s<extra></extra>'), row=1, col=2)
        fig.add_trace(go.Scatter(x=iterations, y=decomp_times, mode='lines', name='Decomp Time', 
                                line=dict(color='red'), hovertemplate='Iteration: %{x}<br>Time: %{y:.6f}s<extra></extra>'), row=2, col=1)
        fig.add_trace(go.Scatter(x=iterations, y=total_times, mode='lines', name='Total Time', 
                                line=dict(color='purple'), hovertemplate='Iteration: %{x}<br>Time: %{y:.6f}s<extra></extra>'), row=2, col=2)
        
        # 平均値の線を追加
        fig.add_hline(y=comm_mean, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {comm_mean:.6f}s", row=1, col=1)
        fig.add_hline(y=comp_mean, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {comp_mean:.6f}s", row=1, col=2)
        fig.add_hline(y=decomp_mean, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {decomp_mean:.6f}s", row=2, col=1)
        fig.add_hline(y=total_mean, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {total_mean:.6f}s", row=2, col=2)
        
        fig.update_layout(title_text="PowerSGD Benchmark - Individual Time Metrics", height=800)
        fig.write_html(os.path.join(output_dir, 'individual_time_metrics_interactive.html'))
        
        # 累積時間メトリクス
        fig = make_subplots(rows=2, cols=2, subplot_titles=('Cumulative Communication Time', 'Cumulative Compression Time', 
                                                           'Cumulative Decompression Time', 'Cumulative Total Processing Time'))
        
        fig.add_trace(go.Scatter(x=iterations, y=comm_cumulative, mode='lines', name='Cumulative Comm Time', 
                                line=dict(color='blue'), hovertemplate='Iteration: %{x}<br>Cumulative Time: %{y:.6f}s<extra></extra>'), row=1, col=1)
        fig.add_trace(go.Scatter(x=iterations, y=comp_cumulative, mode='lines', name='Cumulative Comp Time', 
                                line=dict(color='green'), hovertemplate='Iteration: %{x}<br>Cumulative Time: %{y:.6f}s<extra></extra>'), row=1, col=2)
        fig.add_trace(go.Scatter(x=iterations, y=decomp_cumulative, mode='lines', name='Cumulative Decomp Time', 
                                line=dict(color='red'), hovertemplate='Iteration: %{x}<br>Cumulative Time: %{y:.6f}s<extra></extra>'), row=2, col=1)
        fig.add_trace(go.Scatter(x=iterations, y=total_cumulative, mode='lines', name='Cumulative Total Time', 
                                line=dict(color='purple'), hovertemplate='Iteration: %{x}<br>Cumulative Time: %{y:.6f}s<extra></extra>'), row=2, col=2)
        
        fig.update_layout(title_text="PowerSGD Benchmark - Cumulative Time Metrics", height=800)
        fig.write_html(os.path.join(output_dir, 'cumulative_time_metrics_interactive.html'))
        
        # 統計サマリー（インタラクティブ）
        fig = go.Figure(data=[
            go.Bar(x=metrics, y=means, error_y=dict(type='data', array=stds, visible=True),
                   marker_color=colors, name='Average Time')
        ])
        
        fig.update_layout(
            title='PowerSGD Benchmark - Time Statistics Summary',
            xaxis_title='Metrics',
            yaxis_title='Time (s)',
            height=600
        )
        
        # 値のラベルを追加
        for i, (mean, std) in enumerate(zip(means, stds)):
            fig.add_annotation(
                x=metrics[i], y=mean + std + 0.0001,
                text=f'{mean:.6f}<br>±{std:.6f}',
                yshift=10
            )
        
        fig.write_html(os.path.join(output_dir, 'time_statistics_summary_interactive.html'))
        
    def _create_powerSGD_status_plots(self, epochs, iterations, powerSGD_active_list, current_ranks, output_dir):
        """PowerSGD状態の可視化を作成"""
        import numpy as np
        
        # PowerSGD状態の可視化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('PowerSGD Status and Rank Changes', fontsize=16)
        
        # 上側：PowerSGDの有効/無効状態
        ax1.plot(iterations, powerSGD_active_list, 'b-', alpha=0.7, linewidth=2, label='PowerSGD Active')
        ax1.set_title('PowerSGD Activation Status')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('PowerSGD Active (True/False)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 下側：現在のランク
        ax2.plot(iterations, current_ranks, 'r-', alpha=0.7, linewidth=2, label='Current Rank')
        ax2.set_title('PowerSGD Matrix Approximation Rank')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Rank')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'powerSGD_status.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # インタラクティブ版
        fig = make_subplots(rows=2, cols=1, subplot_titles=('PowerSGD Activation Status', 'PowerSGD Matrix Approximation Rank'))
        
        fig.add_trace(go.Scatter(x=iterations, y=powerSGD_active_list, mode='lines', name='PowerSGD Active', 
                                line=dict(color='blue'), hovertemplate='Iteration: %{x}<br>Active: %{y}<extra></extra>'), row=1, col=1)
        fig.add_trace(go.Scatter(x=iterations, y=current_ranks, mode='lines', name='Current Rank', 
                                line=dict(color='red'), hovertemplate='Iteration: %{x}<br>Rank: %{y}<extra></extra>'), row=2, col=1)
        
        fig.update_layout(title_text="PowerSGD Status and Rank Changes", height=800)
        fig.write_html(os.path.join(output_dir, 'powerSGD_status_interactive.html'))
        
    def _create_memory_plots(self, epochs, iterations, memory_usage, gpu_memory_usage, output_dir):
        """メモリ使用量のグラフを作成"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('PowerSGD Benchmark - Memory Usage', fontsize=16)
        
        # CPUメモリ
        ax1.plot(iterations, memory_usage, 'b-', alpha=0.7)
        ax1.set_title('CPU Memory Usage')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Memory (MB)')
        ax1.grid(True, alpha=0.3)
        
        # GPUメモリ
        ax2.plot(iterations, gpu_memory_usage, 'r-', alpha=0.7)
        ax2.set_title('GPU Memory Usage')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Memory (MB)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'memory_usage.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_compression_plots(self, epochs, iterations, compression_ratios, bandwidths, output_dir):
        """圧縮効率のグラフを作成"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('PowerSGD Benchmark - Compression Efficiency', fontsize=16)
        
        # 圧縮率
        ax1.plot(iterations, compression_ratios, 'g-', alpha=0.7)
        ax1.set_title('Compression Ratio')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Compression Ratio (%)')
        ax1.grid(True, alpha=0.3)
        
        # 通信帯域幅
        ax2.plot(iterations, bandwidths, 'orange', alpha=0.7)
        ax2.set_title('Communication Bandwidth')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Bandwidth (MB/s)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'compression_efficiency.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_learning_plots(self, epochs, iterations, losses, accuracies, output_dir):
        """学習性能のグラフを作成"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('PowerSGD Benchmark - Learning Performance', fontsize=16)
        
        # 損失
        ax1.plot(iterations, losses, 'b-', alpha=0.7)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # 精度
        ax2.plot(iterations, accuracies, 'g-', alpha=0.7)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # _create_learning_plots の先頭に追記して、accuracy列の扱いを変更
    def _create_learning_plots(self, epochs, iterations, losses, accuracies, output_dir):
        """学習性能のグラフを作成"""
        import numpy as np
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('PowerSGD Benchmark - Learning Performance', fontsize=16)

        # 損失（イテレーションごと）
        ax1.plot(iterations, losses, 'b-', alpha=0.7)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)

        # --- Validation Accuracy 可視化 ---
        # 既存の per-iteration accuracies は None/0 が多いので使わない
        # epoch_acc をイテレーション軸に疎サンプルで載せる
        xs, ys = [], []
        for e, it, acc in getattr(self, "epoch_acc", []):
            xs.append(it)
            ys.append(acc)

        # 点とステップ（保持）で見やすく
        if xs:
            # ステップ系列を作る（エポックごとに“保ち”で表示）
            xs_step = []
            ys_step = []
            for i, (x, y) in enumerate(sorted(zip(xs, ys))):
                if i == 0:
                    xs_step.extend([x, x])
                    ys_step.extend([y, y])
                else:
                    # 直前の x から今回 x まで水平に保つ
                    xs_step.append(x)
                    ys_step.append(ys_step[-1])
                    xs_step.append(x)
                    ys_step.append(y)

            ax2.plot(xs_step, ys_step, '-', alpha=0.6, linewidth=2, label='Val Acc (epoch, step)')
            ax2.scatter(xs, ys, s=18, label='Val Acc (epoch points)')
        else:
            # データが無い場合のフォールバック（元の配列をそのまま）
            ax2.plot(iterations, accuracies, 'g-', alpha=0.7, label='Val Acc (raw)')

        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_dashboard(self, epochs, iterations, comm_times, comp_times, decomp_times, 
                         memory_usage, gpu_memory_usage, compression_ratios, bandwidths, 
                         losses, accuracies, output_dir):
        """総合的なダッシュボードを作成"""
        # 統計サマリー
        stats = self.get_summary_stats()
        
        # plotlyでダッシュボード作成
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Communication Time', 'Compression Time', 
                           'Memory Usage', 'Compression Efficiency',
                           'Learning Progress', 'Performance Summary'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # 各サブプロットにデータを追加
        fig.add_trace(go.Scatter(x=iterations, y=comm_times, mode='lines', name='Comm Time'), row=1, col=1)
        fig.add_trace(go.Scatter(x=iterations, y=comp_times, mode='lines', name='Comp Time'), row=1, col=2)
        fig.add_trace(go.Scatter(x=iterations, y=memory_usage, mode='lines', name='CPU Mem'), row=2, col=1)
        fig.add_trace(go.Scatter(x=iterations, y=compression_ratios, mode='lines', name='Comp Ratio'), row=2, col=2)
        fig.add_trace(go.Scatter(x=iterations, y=losses, mode='lines', name='Loss'), row=3, col=1)
        
        # 統計テーブル
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=[
                    ['Total Iterations', 'Avg Comm Time', 'Avg Comp Time', 'Avg Comp Ratio', 'Avg Bandwidth'],
                    [stats.get('total_iterations', 0),
                     f"{stats.get('avg_communication_time', 0):.4f}s",
                     f"{stats.get('avg_compression_time', 0):.4f}s",
                     f"{stats.get('avg_compression_ratio', 0):.2f}%",
                     f"{stats.get('avg_bandwidth_mbps', 0):.2f} MB/s"]
                ],
                fill_color='lavender',
                align='left')
            ),
            row=3, col=2
        )
        
        fig.update_layout(title_text="PowerSGD Benchmark Dashboard", height=1200)
        fig.write_html(os.path.join(output_dir, 'dashboard.html'))
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """統計サマリーを取得"""
        if not self.results:
            return {}
            
        communication_times = [r.communication_time for r in self.results]
        compression_times = [r.compression_time for r in self.results]
        total_times = [r.total_processing_time for r in self.results]
        compression_ratios = [r.compression_ratio for r in self.results]
        bandwidths = [r.bandwidth_mbps for r in self.results]
        
        return {
            'total_iterations': len(self.results),
            'avg_communication_time': sum(communication_times) / len(communication_times),
            'avg_compression_time': sum(compression_times) / len(compression_times),
            'avg_total_time': sum(total_times) / len(total_times),
            'avg_compression_ratio': sum(compression_ratios) / len(compression_ratios),
            'avg_bandwidth_mbps': sum(bandwidths) / len(bandwidths),
            'max_communication_time': max(communication_times),
            'min_communication_time': min(communication_times),
            'max_compression_ratio': max(compression_ratios),
            'min_compression_ratio': min(compression_ratios)
        }
        
    def print_summary(self):
        """統計サマリーを表示"""
        if self.rank != 0:
            return
            
        stats = self.get_summary_stats()
        if not stats:
            print("No benchmark results available.")
            return
            
        print("\n" + "="*50)
        print("PowerSGD Benchmark Summary")
        print("="*50)
        print(f"Total iterations: {stats['total_iterations']}")
        print(f"Average communication time: {stats['avg_communication_time']:.4f}s")
        print(f"Average compression time: {stats['avg_compression_time']:.4f}s")
        print(f"Average total processing time: {stats['avg_total_time']:.4f}s")
        print(f"Average compression ratio: {stats['avg_compression_ratio']:.2f}%")
        print(f"Average bandwidth: {stats['avg_bandwidth_mbps']:.2f} MB/s")
        print(f"Communication time range: {stats['min_communication_time']:.4f}s - {stats['max_communication_time']:.4f}s")
        print(f"Compression ratio range: {stats['min_compression_ratio']:.2f}% - {stats['max_compression_ratio']:.2f}%")
        print("="*50)


def create_powerSGD_hook_with_benchmark(
    benchmark: PowerSGDBenchmark,
    matrix_approximation_rank: int = 1,
    start_powerSGD_iter: int = 100,
    use_error_feedback: bool = True
):
    """ベンチマーク機能付きのPowerSGDフックを作成"""
    
    def powerSGD_hook_with_benchmark(state: PowerSGDState, bucket):
        # 圧縮開始
        benchmark.start_compression()
        
        # 元のサイズを記録
        original_size = bucket.buffer().numel() * bucket.buffer().element_size()
        
        # 元のPowerSGDフックを実行
        result = powerSGD_hook(state, bucket)
        
        # 圧縮終了
        compression_time = benchmark.end_compression()
        
        # 復元開始
        benchmark.start_decompression()
        
        # 復元終了（実際の復元処理はPowerSGDフック内で行われる）
        decompression_time = benchmark.end_decompression()
        
        # 圧縮後のサイズを推定（PowerSGDの行列近似ランクに基づく）
        # PowerSGDでは、勾配行列GをG ≈ PQ^Tの形で近似する
        # 元のサイズ: M × N
        # 圧縮後: M × rank + N × rank = (M + N) × rank
        # 圧縮率: 1 - (M + N) * rank / (M * N)
        # 簡易的に、rank=1の場合の圧縮率を推定
        matrix_rank = state.matrix_approximation_rank
        compressed_size = int(original_size * matrix_rank / 100)  # 簡易的な推定
        
        return result
    
    # PowerSGDStateを作成
    state = PowerSGDState(
        process_group=None,
        matrix_approximation_rank=matrix_approximation_rank,
        start_powerSGD_iter=start_powerSGD_iter,
        use_error_feedback=use_error_feedback
    )
    
    return state, powerSGD_hook_with_benchmark


def measure_network_performance(rank: int, world_size: int, message_sizes: List[int] = None) -> Dict[str, float]:
    """ネットワーク性能を計測"""
    if message_sizes is None:
        message_sizes = [1024, 1024*1024, 10*1024*1024]  # 1KB, 1MB, 10MB
        
    results = {}
    
    # 全プロセスで初期同期
    dist.barrier()
    
    if rank == 0:
        print("Measuring network performance...")
        
        for size in message_sizes:
            tensor_send = torch.ones(size, dtype=torch.float32).cuda()
            tensor_recv = torch.zeros(size, dtype=torch.float32).cuda()
            
            # 各サイズごとに同期
            dist.barrier()
            
            # 複数回計測して平均を取る
            times = []
            for _ in range(5):
                start_time = time.time()
                dist.send(tensor_send, dst=1 if world_size > 1 else 0)
                dist.recv(tensor_recv, src=1 if world_size > 1 else 0)
                end_time = time.time()
                times.append(end_time - start_time)
                
            avg_time = sum(times) / len(times)
            data_mb = (2.0 * size * 4) / (1024.0 * 1024.0)  # float32 = 4 bytes
            bandwidth = data_mb / avg_time if avg_time > 0 else 0.0
            
            results[f"bandwidth_{size}"] = bandwidth
            results[f"latency_{size}"] = avg_time * 1000.0 / 2.0  # ms
            
    elif rank == 1 and world_size > 1:
        for size in message_sizes:
            tensor_send = torch.ones(size, dtype=torch.float32).cuda()
            tensor_recv = torch.zeros(size, dtype=torch.float32).cuda()
            
            # 各サイズごとに同期
            dist.barrier()
            
            for _ in range(5):
                dist.recv(tensor_recv, src=0)
                dist.send(tensor_send, dst=0)
                
    # 最終同期
    dist.barrier()
    return results 
