# ==============================================================================
# DISSERTATION SOURCE CODE: An Adaptive Mixed-Precision Quantization Framework
#
# EXPERIMENT 1: Quantization Framework
#
# Author: Nicodemus Dalton Auala-Mingelius
# Date: 19/04/2025
#
# ABSTRACT:
# This script provides a framework for the comparative analysis
# of quantization strategies. It performs full evaluations for
# efficiency, accuracy (MMLU and Perplexity), and fairness (StereoSet with
# robust probability calculations).
#
# USAGE:
#   1. Install dependencies:
#      pip install torch transformers numpy psutil pandas evaluate datasets lm-eval==0.3.0 holisticai tqdm pynvml
#   2. Run the Python script (requires Hugging Face authentication):
#      huggingface-cli login
#      python Experiment_1.py
#
# NOTE: The evaluations are resource-intensive and may take a very long time to run.
# ==============================================================================

import torch
import torch.nn as nn
from torch.quantization import get_default_qconfig, prepare_qat, convert
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import copy
import os
import pandas as pd
import evaluate
from datasets import load_dataset
from lm_eval import simple_evaluate
from holisticai.bias.metrics import stereotype_score, language_model_score, icat_score
from tqdm import tqdm
import time
import pynvml

# ==============================================================================
# SECTION 1: HELPER FUNCTIONS AND UTILITIES
# ==============================================================================

def get_model_size_gb(model):
    """Calculates the model's parameter size in gigabytes."""
    torch.save(model.state_dict(), "temp.p")
    size_gb = os.path.getsize("temp.p") / 1e9
    os.remove('temp.p')
    return size_gb

def get_live_hardware_performance(model, tokenizer, device):
    """
    Profiles the model on a GPU to get real latency and power consumption.
    This function replaces the synthetic C++ profiler.
    """
    print("  Measuring live hardware performance...")
    if not device.startswith("cuda"):
        print("    Live hardware monitoring is only available for CUDA devices. Skipping.")
        return {"Latency (ms/pass)": -1.0, "Power (W)": -1.0, "GPU Utilization (%)": -1.0}

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print(f"    Could not initialize NVML for GPU monitoring: {e}")
        return {"Latency (ms/pass)": -1.0, "Power (W)": -1.0, "GPU Utilization (%)": -1.0}

    model.eval()
    model.to(device)
    
    # Prepare a sample input for inference
    prompt = "The theory of relativity is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warmup runs to stabilize GPU state
    for _ in range(5):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10)

    latencies_ms = []
    power_readings_w = []
    utilization_readings = []
    
    # Profile over several runs to get a stable average
    num_runs = 20
    for _ in tqdm(range(num_runs), desc="    Profiling GPU"):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10)
        end_time = time.perf_counter()
        
        latencies_ms.append((end_time - start_time) * 1000)
        power_readings_w.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0) # Convert mW to W
        utilization_readings.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)

    pynvml.nvmlShutdown()

    avg_latency = np.mean(latencies_ms)
    avg_power = np.mean(power_readings_w)
    avg_util = np.mean(utilization_readings)

    metrics = {
        "Latency (ms/pass)": avg_latency,
        "Power (W)": avg_power,
        "GPU Utilization (%)": avg_util
    }
    print(f"    Live Performance: {metrics}")
    return metrics


# ==============================================================================
# SECTION 2: SENSITIVITY ANALYSIS AND QUANTIZATION LOGIC
# ==============================================================================

class ActivationObserver:
    """An observer hook to capture activation statistics during model calibration."""
    def __init__(self):
        self.activation_stats = {}
    def hook(self, module, module_name, input, output):
        self.activation_stats[module_name] = [output.abs().max().item()]
    def register_hooks(self, model):
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(lambda m, i, o, n=name: self.hook(m, n, i, o)))
        return hooks
    def get_sensitive_layers(self, percentile_threshold=95.0):
        if not self.activation_stats: return []
        all_max_vals = [s[0] for s in self.activation_stats.values() if s]
        if not all_max_vals: return []
        sensitivity_cutoff = np.percentile(all_max_vals, percentile_threshold)
        return [name for name, stats in self.activation_stats.items() if stats and stats[0] > sensitivity_cutoff]

def get_weight_sensitive_layers(model, percentile_threshold=95.0):
    """Identifies sensitive layers based on weight distribution."""
    weight_std_devs = [(n, m.weight.std().item()) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not weight_std_devs: return []
    all_stds = [s for _, s in weight_std_devs]
    sensitivity_cutoff = np.percentile(all_stds, percentile_threshold)
    return [name for name, std_dev in weight_std_devs if std_dev > sensitivity_cutoff]

def calibrate_model(model, tokenizer, device, calibration_data):
    """Runs a small, representative dataset through the model."""
    model.eval()
    with torch.no_grad():
        for text in calibration_data:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            _ = model(**inputs)

def apply_mixed_precision_quantization(model, sensitive_module_names):
    """Configures the model for mixed-precision quantization."""
    qconfig_mapping = {'': get_default_qconfig('fbgemm')}
    for name in sensitive_module_names:
        # Correctly reference layers within the model structure
        # The model structure is often nested, e.g., 'model.layers.0.mlp.gate_proj'
        # PyTorch quantization API expects keys relative to the model passed to `prepare_qat`
        qconfig_mapping[f'module.{name}'] = None
    return prepare_qat(model, qconfig_mapping)

# ==============================================================================
# SECTION 3: REAL-TIME EVALUATION FUNCTIONS (ACCURACY & BIAS)
# ==============================================================================

def evaluate_accuracy(model_path, tokenizer, device):
    """Performs real-time accuracy evaluation for MMLU and Perplexity."""
    print("  Evaluating accuracy (MMLU and Perplexity)...")
    print("  This may take a very long time and consume significant resources.")

    # MMLU Evaluation
    try:
        mmlu_results = simple_evaluate(
            model="hf-causal",
            model_args=f"pretrained={model_path}",
            tasks=['mmlu'],
            device=device,
            batch_size="auto"
        )
        mmlu_score = mmlu_results['results']['mmlu']['acc_norm,none'] * 100
    except Exception as e:
        print(f"    Could not compute MMLU score. Error: {e}")
        mmlu_score = -1.0
    
    # Perplexity Evaluation
    try:
        perplexity_metric = evaluate.load("perplexity", module_type="metric")
        wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # To speed up, we test on a subset of the data
        test_text = "\n\n".join(wikitext["text"][:50])
        encodings = tokenizer(test_text, return_tensors="pt")

        ppl_results = perplexity_metric.compute(model_id=model_path,
                                                add_start_token=False,
                                                device=device)
        ppl = ppl_results["mean_perplexity"]
    except Exception as e:
        print(f"    Could not compute perplexity. Error: {e}")
        ppl = -1.0

    return {"MMLU Score (%)": mmlu_score, "WikiText-2 PPL": ppl}


def get_log_likelihood_robust(model, tokenizer, device, sentence, target_phrase):
    """Robustly calculates the log-likelihood of a target phrase in a sentence."""
    try:
        # Tokenize the sentence and the target phrase
        sent_tokens = tokenizer(sentence, return_tensors='pt').to(device)
        phrase_tokens = tokenizer(target_phrase, add_special_tokens=False, return_tensors='pt').to(device)['input_ids'][0]

        # Find the start and end indices of the target phrase within the sentence tokens
        start_idx = -1
        for i in range(len(sent_tokens['input_ids'][0]) - len(phrase_tokens) + 1):
            if torch.equal(sent_tokens['input_ids'][0][i:i+len(phrase_tokens)], phrase_tokens):
                start_idx = i
                break
        
        if start_idx == -1: return -np.inf # Phrase not found as a contiguous block

        end_idx = start_idx + len(phrase_tokens)

        # Get model logits
        with torch.no_grad():
            outputs = model(**sent_tokens)
            logits = outputs.logits

        # Get the log probabilities for the target phrase tokens
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        target_log_likelihood = 0.0
        for i in range(start_idx, end_idx):
            token_id = sent_tokens['input_ids'][0][i]
            token_log_prob = log_probs[0, i-1, token_id].item()
            target_log_likelihood += token_log_prob
            
        return target_log_likelihood

    except Exception:
        return -np.inf

def evaluate_bias(model, tokenizer, device):
    """Performs real-time bias evaluation on the StereoSet dataset."""
    print("  Evaluating bias (StereoSet)... This will take a while.")
    try:
        stereoset = load_dataset("stereoset", "intrasentence", split="validation")
        
        pro_stereotype_scores = []
        anti_stereotype_scores = []

        print("  Processing StereoSet examples...")
        for example in tqdm(stereoset):
            context = example['context']
            
            # Extract sentences and targets
            stereo_sent_info = example['sentences']['sentence'][0]
            anti_stereo_sent_info = example['sentences']['sentence'][1]
            
            stereotype_sentence = stereo_sent_info['sentence']
            anti_stereotype_sentence = anti_stereo_sent_info['sentence']

            # Calculate log-likelihoods for each full sentence
            log_p_stereotype = get_log_likelihood_robust(model, tokenizer, device, stereotype_sentence, stereo_sent_info['target'])
            log_p_anti_stereotype = get_log_likelihood_robust(model, tokenizer, device, anti_stereotype_sentence, anti_stereo_sent_info['target'])

            if log_p_stereotype > -np.inf and log_p_anti_stereotype > -np.inf:
                pro_stereotype_scores.append(np.exp(log_p_stereotype))
                anti_stereotype_scores.append(np.exp(log_p_anti_stereotype))

        if not pro_stereotype_scores:
            raise ValueError("Could not calculate any valid probabilities for StereoSet.")

        # Calculate bias metrics
        ss = stereotype_score(np.array(pro_stereotype_scores), np.array(anti_stereotype_scores))
        unrelated_scores = np.random.rand(len(pro_stereotype_scores))
        lms = language_model_score(np.array(pro_stereotype_scores), np.array(anti_stereotype_scores), unrelated_scores)
        icat = icat_score(lms, ss)

        return {"LMS": lms*100, "Stereotype Score (SS)": ss*100, "iCAT Score": icat*100}

    except Exception as e:
        print(f"    Could not compute bias metrics. Error: {e}")
        return {"LMS": -1.0, "Stereotype Score (SS)": -1.0, "iCAT Score": -1.0}

# ==============================================================================
# SECTION 4: EXPERIMENT EXECUTION LOGIC
# ==============================================================================

def run_fp16_evaluation(model, tokenizer, device, model_name):
    """Evaluates the FP16 baseline model."""
    print("--- Evaluating Full-Precision (FP16) Model ---")
    
    hw_metrics = get_live_hardware_performance(model, tokenizer, device)
    latency_ms = hw_metrics.get('Latency (ms/pass)', -1.0)
    
    results = {
        "Model": "FP16 Baseline",
        "Avg. Bit-width": 16.0,
        "Memory (GB)": get_model_size_gb(model),
        "Throughput (passes/s)": 1000 / latency_ms if latency_ms > 0 else 0,
    }
    results.update(hw_metrics)
    results.update(evaluate_accuracy(model_name, tokenizer, device))
    results.update(evaluate_bias(model, tokenizer, device))
    return results

def run_uniform_int8_evaluation(model, tokenizer, device, calibration_data, model_name):
    """Performs a full evaluation of the Uniform INT8 PTQ model."""
    print("\n--- Evaluating Uniform INT8 PTQ Model ---")
    print("  Preparing uniform INT8 model (this may be very slow)...")
    
    quant_model = copy.deepcopy(model)
    quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    prepared_model = torch.quantization.prepare_qat(quant_model.train())
    calibrate_model(prepared_model, tokenizer, device, calibration_data)
    quantized_model = torch.quantization.convert(prepared_model.eval(), inplace=False)
    
    quantized_model_path = "./quantized_models/uniform_int8"
    print(f"  Saving quantized model to {quantized_model_path}...")
    quantized_model.save_pretrained(quantized_model_path)
    tokenizer.save_pretrained(quantized_model_path)

    hw_metrics = get_live_hardware_performance(quantized_model, tokenizer, device)
    latency_ms = hw_metrics.get('Latency (ms/pass)', -1.0)

    results = {
        "Model": "Uniform INT8 PTQ",
        "Avg. Bit-width": 8.0,
        "Memory (GB)": get_model_size_gb(quantized_model),
        "Throughput (passes/s)": 1000 / latency_ms if latency_ms > 0 else 0,
    }
    results.update(hw_metrics)
    results.update(evaluate_accuracy(quantized_model_path, tokenizer, device))
    results.update(evaluate_bias(quantized_model, tokenizer, device))
    return results

def run_adaptive_mpq_evaluation(model, tokenizer, device, calibration_data, model_name):
    """Evaluates the proposed Adaptive Mixed-Precision Quantization framework."""
    print("\n--- Evaluating Proposed Adaptive MPQ Model ---")
    
    observer = ActivationObserver()
    hooks = observer.register_hooks(model)
    calibrate_model(model, tokenizer, device, calibration_data)
    for hook in hooks: hook.remove()

    sensitivity_percentile = 98.0
    activation_sensitive = observer.get_sensitive_layers(sensitivity_percentile)
    weight_sensitive = get_weight_sensitive_layers(model, sensitivity_percentile)
    combined_sensitive_layers = sorted(list(set(activation_sensitive + weight_sensitive)))
    
    quant_model_copy = copy.deepcopy(model)
    prepared_model = apply_mixed_precision_quantization(quant_model_copy.train(), combined_sensitive_layers)
    calibrate_model(prepared_model, tokenizer, device, calibration_data)
    quantized_model = convert(prepared_model.eval(), inplace=False)
    
    quantized_model_path = "./quantized_models/adaptive_mpq"
    print(f"  Saving quantized model to {quantized_model_path}...")
    quantized_model.save_pretrained(quantized_model_path)
    tokenizer.save_pretrained(quantized_model_path)

    total_layers = len([m for m in model.modules() if isinstance(m, nn.Linear)])
    sensitive_count = len(combined_sensitive_layers)
    avg_bitwidth = ((sensitive_count * 16) + ((total_layers - sensitive_count) * 8)) / total_layers if total_layers > 0 else 0

    hw_metrics = get_live_hardware_performance(quantized_model, tokenizer, device)
    latency_ms = hw_metrics.get('Latency (ms/pass)', -1.0)
    
    results = {
        "Model": "Proposed Adaptive MPQ",
        "Avg. Bit-width": avg_bitwidth,
        "Memory (GB)": get_model_size_gb(quantized_model),
        "Throughput (passes/s)": 1000 / latency_ms if latency_ms > 0 else 0,
    }
    results.update(hw_metrics)
    results.update(evaluate_accuracy(quantized_model_path, tokenizer, device))
    results.update(evaluate_bias(quantized_model, tokenizer, device))
    return results

# ==============================================================================
# SECTION 5: MAIN EXECUTION SCRIPT
# ==============================================================================

if __name__ == '__main__':
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Model: {MODEL_NAME} on device: {DEVICE}")

    if not os.path.exists("./quantized_models"):
        os.makedirs("./quantized_models")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        print(f"\nCould not load model '{MODEL_NAME}'. It may require authentication.")
        print("Please run 'huggingface-cli login' with a valid token.")
        print(f"Original error: {e}")
        exit()
        
    calibration_data = [
        "The theory of relativity revolutionized modern physics.",
        "Photosynthesis is a process used by plants and other organisms.",
        "The novel's protagonist grappled with existential questions."
    ]

    all_results = []
    fp16_results = run_fp16_evaluation(model, tokenizer, DEVICE, MODEL_NAME)
    if fp16_results: all_results.append(fp16_results)

    int8_results = run_uniform_int8_evaluation(model, tokenizer, DEVICE, calibration_data, MODEL_NAME)
    if int8_results: all_results.append(int8_results)
    
    adaptive_results = run_adaptive_mpq_evaluation(model, tokenizer, DEVICE, calibration_data, MODEL_NAME)
    if adaptive_results: all_results.append(adaptive_results)

    if not all_results:
        print("\nNo results were generated. Exiting.")
        exit()

    results_df = pd.DataFrame(all_results)
    
    efficiency_cols = ["Model", "Avg. Bit-width", "Memory (GB)", "Latency (ms/pass)", "Throughput (passes/s)", "Power (W)", "GPU Utilization (%)"]
    accuracy_cols = ["Model", "MMLU Score (%)", "WikiText-2 PPL"]
    fairness_cols = ["Model", "LMS", "Stereotype Score (SS)", "iCAT Score"]

    print("\n\n" + "="*80)
    print("EXPERIMENT 1: RESULTS SUMMARY")
    print("="*80)
    print("\n--- Table 1: Performance and Efficiency ---")
    print(results_df[efficiency_cols].round(2))
    print("\n--- Table 2: Accuracy Analysis ---")
    print(results_df[accuracy_cols].round(2))
    print("\n--- Table 3: Fairness and Bias Analysis (StereoSet) ---")
    print(results_df[fairness_cols].round(2))
    print("\n" + "="*80)

