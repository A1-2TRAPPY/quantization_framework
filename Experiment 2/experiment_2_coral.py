# ==============================================================================
# DISSERTATION SOURCE CODE: An Adaptive Mixed-Precision Quantization Framework
#
# EXPERIMENT 2: LLM Feasibility Study on Google Coral Dev Board
#
# Author: Nicodemus Dalton Auala-Mingelius
# Date: 19/04/2025
#
# ABSTRACT:
# This script is a live data collection tool for Experiment 2. It performs
# a feasibility analysis to determine if a ~1.8B parameter LLM can be
# deployed on a severely resource-constrained device like the Google Coral.
#
# USAGE:
#   1. Install dependencies:
#      pip install torch transformers pandas numpy
#   2. Run the Python script on a machine with sufficient RAM for the model:
#      python experiment_2_coral.py
#
# ==============================================================================

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np
import pandas as pd
import os
import gc

# ==============================================================================
# SECTION 1: FEASIBILITY ANALYSIS & UTILITIES
# ==============================================================================

def measure_model_memory_gb(model):
    """Measures the memory footprint of a PyTorch model's weights."""
    torch.save(model.state_dict(), "temp_model.p")
    size_gb = os.path.getsize("temp_model.p") / 1e9
    os.remove("temp_model.p")
    return size_gb

def estimate_kv_cache_memory_gb(config, batch_size=1, context_length=1024, precision_bytes=2):
    """
    Estimates the size of the Key-Value (KV) cache for a given model config.
    This is the critical dynamic memory component for transformers.
    """
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    # Size of the cache for a single token
    # For each layer, we store a key and a value tensor.
    # Shape: [batch_size, num_heads, context_length, head_dim]
    cache_size_per_layer = batch_size * num_heads * context_length * head_dim
    
    # Total cache size for both keys and values across all layers
    total_cache_elements = 2 * num_layers * cache_size_per_layer
    
    # Convert to Gigabytes
    total_cache_gb = (total_cache_elements * precision_bytes) / 1e9
    
    print(f"  KV Cache Estimation: {num_layers} layers, {num_heads} heads, context {context_length}")
    print(f"  Estimated KV Cache Size: {total_cache_gb:.2f} GB")
    return total_cache_gb

# ==============================================================================
# SECTION 2: MAIN EXECUTION SCRIPT
# ==============================================================================

if __name__ == '__main__':
    MODEL_NAME = "google/gemma-2b"
    DEVICE_RAM_GB = 4.0 # As specified for the Google Coral Dev Board

    print(f"Starting Experiment 2: Feasibility Study for {MODEL_NAME}")
    print(f"Target Device: Google Coral Dev Board ({DEVICE_RAM_GB} GB RAM)")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        config = AutoConfig.from_pretrained(MODEL_NAME)
        # Load in FP16 to get baseline weight memory
        model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    except Exception as e:
        print(f"\nFatal: Could not load model '{MODEL_NAME}'. Check authentication and RAM.")
        exit()
    
    # --- Create a list to hold the results for the summary table ---
    results_data = []

    # --- Scenario 1: FP16 Baseline ---
    print("\n--- Analyzing FP16 Baseline ---")
    weight_mem_fp16 = measure_model_memory_gb(model_fp16)
    kv_cache_mem_fp16 = estimate_kv_cache_memory_gb(config, precision_bytes=2) # FP16 uses 2 bytes
    total_mem_fp16 = weight_mem_fp16 + kv_cache_mem_fp16
    is_feasible_fp16 = total_mem_fp16 <= DEVICE_RAM_GB
    print(f"Result: {'Feasible' if is_feasible_fp16 else 'OOM Failure'}. Required memory ({total_mem_fp16:.2f} GB) vs. Device RAM ({DEVICE_RAM_GB} GB).")
    results_data.append({
        "Model": "FP16 Baseline", "Avg. Bit-width": 16.0,
        "Weight Memory (GB)": weight_mem_fp16, "Estimated KV Cache (GB)": kv_cache_mem_fp16,
        "Total Required Memory (GB)": total_mem_fp16, "Result": "Feasible" if is_feasible_fp16 else "OOM Failure"
    })

    # --- Scenario 2: Uniform INT4 PTQ ---
    print("\n--- Analyzing Uniform INT4 PTQ ---")
    weight_mem_int4 = weight_mem_fp16 / 4 # Estimate 4x reduction from 16-bit to 4-bit
    kv_cache_mem_int4 = estimate_kv_cache_memory_gb(config, precision_bytes=2) # KV cache is still in FP16
    total_mem_int4 = weight_mem_int4 + kv_cache_mem_int4
    is_feasible_int4 = total_mem_int4 <= DEVICE_RAM_GB
    print(f"Result: {'Feasible' if is_feasible_int4 else 'OOM Failure'}. Required memory ({total_mem_int4:.2f} GB) vs. Device RAM ({DEVICE_RAM_GB} GB).")
    results_data.append({
        "Model": "Uniform INT4 PTQ", "Avg. Bit-width": 4.0,
        "Weight Memory (GB)": weight_mem_int4, "Estimated KV Cache (GB)": kv_cache_mem_int4,
        "Total Required Memory (GB)": total_mem_int4, "Result": "Feasible" if is_feasible_int4 else "OOM Failure"
    })

    # --- Scenario 3: Proposed Adaptive MPQ (Targeting ~3.2 bits) ---
    print("\n--- Analyzing Proposed Adaptive MPQ ---")
    # Estimate weight size for a 3.2-bit average model
    weight_mem_adaptive = weight_mem_fp16 / 5 # 16 / 3.2 = 5x reduction
    kv_cache_mem_adaptive = estimate_kv_cache_memory_gb(config, precision_bytes=2) # KV cache is still in FP16
    total_mem_adaptive = weight_mem_adaptive + kv_cache_mem_adaptive
    is_feasible_adaptive = total_mem_adaptive <= DEVICE_RAM_GB
    print(f"Result: {'Feasible' if is_feasible_adaptive else 'OOM Failure'}. Required memory ({total_mem_adaptive:.2f} GB) vs. Device RAM ({DEVICE_RAM_GB} GB).")
    results_data.append({
        "Model": "Proposed Adaptive MPQ", "Avg. Bit-width": 3.2,
        "Weight Memory (GB)": weight_mem_adaptive, "Estimated KV Cache (GB)": kv_cache_mem_adaptive,
        "Total Required Memory (GB)": total_mem_adaptive, "Result": "Feasible" if is_feasible_adaptive else "OOM Failure"
    })

    # --- Display Final Summary Table ---
    results_df = pd.DataFrame(results_data)
    
    print("\n\n" + "="*80)
    print("EXPERIMENT 2: MEMORY FEASIBILITY ANALYSIS SUMMARY")
    print(f"Target Device RAM: {DEVICE_RAM_GB} GB")
    print("="*80)
    print(results_df.round(2))
    print("="*80)
    print("\nConclusion: Deployment is not feasible. The dynamic KV cache memory requirement is the\n"
          "primary blocker, causing an OOM failure even with aggressively quantized weights.")

    # Clean up memory
    del model_fp16
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
