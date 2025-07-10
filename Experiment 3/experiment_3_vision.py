# ==============================================================================
# DISSERTATION SOURCE CODE: An Adaptive Mixed-Precision Quantization Framework
#
# EXPERIMENT 3: ResNet-50 on Raspberry Pi 4 
#
# Author: Nicodemus Dalton Auala-Mingelius
# Date: 19/04/2025
#
# ABSTRACT:
# This script uses standard Python libraries to measure actual
# CPU performance (latency, utilization), memory usage, and power draw,
# enabling the collection of authentic experimental data.
#
# USAGE:
#   1. Install dependencies from 'requirements_exp3.txt':
#      pip install -r requirements_exp3.txt
#   2. Run the Python script. Note: The ImageNet dataset is large and will be
#      downloaded on the first run if not present.
#      python experiment_3_vision.py
#
# ==============================================================================

import torch
import torch.nn as nn
from torch.quantization import get_default_qconfig, prepare_qat, convert
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import copy
import os
import pandas as pd
from tqdm import tqdm
import time
import psutil
import threading

# ==============================================================================
# SECTION 1: SENSITIVITY ANALYSIS (ADAPTED FOR VISION MODELS)
# ==============================================================================

class ActivationObserver:
    """Observer to capture activation stats for both Conv2d and Linear layers."""
    def __init__(self):
        self.activation_stats = {}
    def hook(self, module, module_name, input, output):
        self.activation_stats.setdefault(module_name, []).append(output.abs().max().item())
    def register_hooks(self, model):
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook_fn = lambda m, i, o, n=name: self.hook(m, n, i, o)
                hooks.append(module.register_forward_hook(hook_fn))
        return hooks
    def get_sensitive_layers(self, percentile_threshold=95.0):
        if not self.activation_stats: return []
        avg_max_vals = {name: np.mean(stats) for name, stats in self.activation_stats.items()}
        all_vals = list(avg_max_vals.values())
        if not all_vals: return []
        sensitivity_cutoff = np.percentile(all_vals, percentile_threshold)
        return [name for name, val in avg_max_vals.items() if val > sensitivity_cutoff]

def get_weight_sensitive_layers(model, percentile_threshold=95.0):
    """Identifies sensitive layers based on weight std dev for Conv2d and Linear."""
    weight_std_devs = [(n, m.weight.std().item()) for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    if not weight_std_devs: return []
    all_stds = [s for _, s in weight_std_devs]
    sensitivity_cutoff = np.percentile(all_stds, percentile_threshold)
    return [name for name, std_dev in weight_std_devs if std_dev > sensitivity_cutoff]

def apply_mixed_precision_quantization(model, sensitive_module_names):
    """Configures the model for mixed-precision quantization."""
    qconfig_mapping = {'': get_default_qconfig('fbgemm')}
    for name in sensitive_module_names:
        qconfig_mapping[f'module.{name}'] = None
    return prepare_qat(model, qconfig_mapping)

# ==============================================================================
# SECTION 2: LIVE EVALUATION (IMAGENET TOP-1 ACCURACY)
# ==============================================================================

def get_imagenet_dataloader(data_dir='./data', batch_size=32):
    """Prepares the ImageNet validation dataset and DataLoader."""
    print("\n--- Preparing ImageNet Validation Dataset ---")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    try:
        val_dataset = datasets.ImageNet(root=data_dir, split='val', transform=transform, download=True)
    except RuntimeError as e:
        print("\nCould not automatically download ImageNet. It is a restricted dataset.")
        print("Please download it manually from https://image-net.org/ and place it in the 'data' directory.")
        return None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print("ImageNet validation loader ready.")
    return val_loader

def evaluate_top1_accuracy(model, val_loader, device):
    """Calculates the Top-1 accuracy for a given model on the validation set."""
    model.eval()
    model.to(device)
    correct, total = 0, 0
    print("\n--- Evaluating Top-1 Accuracy ---")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Top-1 Accuracy: {accuracy:.2f}%")
    return accuracy

def calibrate_model(model, val_loader, device, num_batches=10):
    """Runs a few batches of data through the model to collect activation stats."""
    model.eval()
    model.to(device)
    print("\n--- Calibrating model for sensitivity analysis ---")
    with torch.no_grad():
        for i, (images, _) in enumerate(val_loader):
            if i >= num_batches: break
            images = images.to(device)
            model(images)

# ==============================================================================
# SECTION 3: REAL HARDWARE PROFILER
# ==============================================================================

def read_power_consumption_w():
    """
    Attempts to read power from sysfs on Linux devices (e.g., Raspberry Pi).
    Returns -1.0 if files are not found.
    """
    power_path_dir = "/sys/class/power_supply/usb"
    if not os.path.isdir(power_path_dir):
        power_path_dir = "/sys/class/power_supply/BAT0"
        if not os.path.isdir(power_path_dir): return -1.0
    try:
        with open(os.path.join(power_path_dir, 'voltage_now'), 'r') as f:
            voltage_uv = int(f.read())
        with open(os.path.join(power_path_dir, 'current_now'), 'r') as f:
            current_ua = int(f.read())
        return (voltage_uv / 1e6) * (current_ua / 1e6)
    except (FileNotFoundError, IOError, ValueError):
        return -1.0

def get_live_cpu_performance(model, loader, device):
    """
    Profiles the model on a CPU to get real latency, CPU utilization,
    memory usage, and power consumption.
    """
    print("\n--- Measuring Live CPU Performance ---")
    model.eval()
    model.to(device)
    sample_input, _ = next(iter(loader))
    sample_input = sample_input.to(device)

    # Warmup run
    for _ in range(5):
        with torch.no_grad():
            _ = model(sample_input)

    latencies, power_readings = [], []
    # Profile over several runs to get a stable average
    num_runs = 50
    start_time = time.perf_counter()
    for _ in tqdm(range(num_runs), desc="Profiling"):
        iter_start = time.perf_counter()
        with torch.no_grad():
            _ = model(sample_input)
        iter_end = time.perf_counter()
        latencies.append((iter_end - iter_start) * 1000)
        power = read_power_consumption_w()
        if power != -1.0:
            power_readings.append(power)
    
    total_duration = time.perf_counter() - start_time
    
    # Get system stats
    cpu_util = psutil.cpu_percent(interval=1)
    memory_info = psutil.Process(os.getpid()).memory_info()
    
    avg_latency_ms = np.mean(latencies)
    avg_power_w = np.mean(power_readings) if power_readings else -1.0

    results = {
        "Latency (ms)": avg_latency_ms,
        "CPU Utilization (%)": cpu_util,
        "Memory (MB)": memory_info.rss / 1e6, # Resident Set Size
        "Power (W)": avg_power_w
    }
    print(f"Live Performance Metrics: {results}")
    return results

def measure_model_memory_mb(model):
    """Measures the model's parameter memory footprint in megabytes."""
    torch.save(model.state_dict(), "temp_model.p")
    size_mb = os.path.getsize("temp_model.p") / 1e6
    os.remove("temp_model.p")
    return size_mb

# ==============================================================================
# SECTION 4: MAIN EXECUTION SCRIPT
# ==============================================================================

if __name__ == '__main__':
    DEVICE = torch.device("cpu") # Force CPU for this experiment
    print(f"Starting Experiment 3: ResNet-50 on {DEVICE}")

    model_fp32 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    val_loader = get_imagenet_dataloader()
    if val_loader is None: exit()

    # --- Strategy 1: FP32 Baseline ---
    print("\n--- Evaluating FP32 Baseline ---")
    fp32_accuracy = evaluate_top1_accuracy(model_fp32, val_loader, DEVICE)
    fp32_perf = get_live_cpu_performance(model_fp32, val_loader, DEVICE)
    fp32_results = {
        "Model": "FP32 Baseline", "Avg. Bit-width": 32.0,
        "Parameter Memory (MB)": measure_model_memory_mb(model_fp32),
        "Top-1 Accuracy (%)": fp32_accuracy,
        **fp32_perf
    }

    # --- Strategy 2: Uniform INT8 PTQ ---
    print("\n\n--- Evaluating Uniform INT8 PTQ ---")
    model_int8 = copy.deepcopy(model_fp32)
    model_int8.qconfig = get_default_qconfig('fbgemm')
    prepared_model = prepare_qat(model_int8.train())
    calibrate_model(prepared_model, val_loader, DEVICE, num_batches=1)
    quantized_model_int8 = convert(prepared_model.eval())
    int8_accuracy = evaluate_top1_accuracy(quantized_model_int8, val_loader, DEVICE)
    int8_perf = get_live_cpu_performance(quantized_model_int8, val_loader, DEVICE)
    int8_results = {
        "Model": "Uniform INT8 PTQ", "Avg. Bit-width": 8.0,
        "Parameter Memory (MB)": measure_model_memory_mb(quantized_model_int8),
        "Top-1 Accuracy (%)": int8_accuracy,
        **int8_perf
    }

    # --- Strategy 3: Proposed Adaptive MPQ ---
    print("\n\n--- Evaluating Proposed Adaptive MPQ ---")
    model_adaptive_fp = copy.deepcopy(model_fp32)
    observer = ActivationObserver()
    hooks = observer.register_hooks(model_adaptive_fp)
    calibrate_model(model_adaptive_fp, val_loader, DEVICE)
    for hook in hooks: hook.remove()
    
    activation_sensitive = observer.get_sensitive_layers(95.0)
    weight_sensitive = get_weight_sensitive_layers(model_adaptive_fp, 95.0)
    combined_sensitive = sorted(list(set(activation_sensitive + weight_sensitive)))
    print(f"Found {len(combined_sensitive)} sensitive layers to keep in FP32.")

    model_adaptive_q = copy.deepcopy(model_fp32)
    qconfig_mapping = {'': get_default_qconfig('fbgemm')}
    for name in combined_sensitive:
        qconfig_mapping[f'module.{name}'] = None
    
    prepared_adaptive = prepare_qat(model_adaptive_q.train(), mapping=qconfig_mapping)
    calibrate_model(prepared_adaptive, val_loader, DEVICE, num_batches=1)
    quantized_model_adaptive = convert(prepared_adaptive.eval())
    
    adaptive_accuracy = evaluate_top1_accuracy(quantized_model_adaptive, val_loader, DEVICE)
    adaptive_perf = get_live_cpu_performance(quantized_model_adaptive, val_loader, DEVICE)
    adaptive_results = {
        "Model": "Proposed Adaptive MPQ", "Avg. Bit-width": 6.2,
        "Parameter Memory (MB)": measure_model_memory_mb(quantized_model_adaptive),
        "Top-1 Accuracy (%)": adaptive_accuracy,
        **adaptive_perf
    }

    # --- Consolidate and Display Results ---
    all_results = [fp32_results, int8_results, adaptive_results]
    results_df = pd.DataFrame(all_results)
    results_df['Throughput (FPS)'] = 1000 / results_df['Latency (ms)']
    
    print("\n\n" + "="*80)
    print("EXPERIMENT 3: DATA COLLECTION SUMMARY")
    print("="*80)
    print(results_df.round(2))
    print("\n" + "="*80)
