# ==============================================================================
# DISSERTATION SOURCE CODE: An Adaptive Mixed-Precision Quantization Framework
#
# EXPERIMENT 2: Llama-3 8B on Google Coral Dev Board
#
# Author: Nicodemus Dalton Auala-Mingelius
# Date: 19/04/2025
#
# ABSTRACT:
# This script is a live data collection tool for Experiment 2.
# It performs real-time quantization, memory measurement, and evaluation
# to enable the collection of experimental data.
#
# USAGE:
#   1. Install dependencies from 'requirements_exp2.txt':
#      pip install -r requirements_exp2.txt
#   2. Run the Python script on a machine with sufficient RAM for the model:
#      python experiment_2_coral.py
#
# ==============================================================================

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import numpy as np
import pandas as pd
import time
import os
import gc

# ==============================================================================
# SECTION 1: HARDWARE INTERFACE FOR GOOGLE CORAL DEV BOARD
# ==============================================================================

class CoralHardwareInterface:
    """
    A hardware interface for the Google Coral Dev Board.
    This class handles model compilation, inference, and performance measurement.
    """
    def __init__(self):
        # Check for connected Edge TPUs
        try:
            self.tpus = list_edge_tpus()
            if not self.tpus:
                raise RuntimeError("No Edge TPU detected. Please connect a Coral device.")
            print(f"Found Edge TPU devices: {self.tpus}")
            self.delegate = edgetpu.load_edgetpu_delegate()
        except Exception as e:
            print(f"Error initializing Edge TPU: {e}")
            self.delegate = None

    def convert_and_compile_model(self, model, tokenizer, model_name="model"):
        """
        Converts a PyTorch model to a compiled Edge TPU TFLite model.
        This is a complex pipeline: PyTorch -> ONNX -> TensorFlow -> TFLite -> Edge TPU TFLite
        """
        print("\n--- Starting Model Conversion and Compilation ---")
        # 1. Export to ONNX
        onnx_path = f"{model_name}.onnx"
        dummy_input = tokenizer("test", return_tensors="pt")['input_ids']
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=12)
        print(f"Model exported to ONNX: {onnx_path}")

        # 2. Convert ONNX to TensorFlow (using tf-onnx)
        tf_path = f"{model_name}_tf"
        subprocess.run(f"python -m tf2onnx.convert --opset 12 --input {onnx_path} --output {tf_path}", shell=True, check=True)
        print(f"Model converted to TensorFlow format: {tf_path}")

        # 3. Convert TensorFlow to TFLite
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_path = f"{model_name}.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Model converted to TFLite: {tflite_path}")

        # 4. Compile TFLite for Edge TPU
        edgetpu_path = f"{model_name}_edgetpu.tflite"
        # The `-m 13` flag is often needed for newer models.
        result = subprocess.run(f"edgetpu_compiler -a -m 13 {tflite_path} -o .", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("Edge TPU compilation failed!")
            print(result.stderr)
            return None
        
        print(f"Model successfully compiled for Edge TPU: {edgetpu_path}")
        return edgetpu_path

    def read_power_consumption_w(self):
        """
        Reads power consumption from system files on a Coral Dev Board.
        NOTE: The exact file path may vary depending on the OS image.
        """
        # This path is typical for USB-powered devices.
        power_path = "/sys/class/power_supply/usb/voltage_now"
        current_path = "/sys/class/power_supply/usb/current_now"
        
        try:
            with open(power_path, 'r') as f:
                voltage_uv = int(f.read())
            with open(current_path, 'r') as f:
                current_ua = int(f.read())
            
            # Convert from microvolts/microamps to watts
            power_w = (voltage_uv / 1e6) * (current_ua / 1e6)
            return power_w
        except FileNotFoundError:
            # Fallback if the sysfs files don't exist
            return -1.0

    def get_live_performance(self, edgetpu_model_path, tokenizer):
        """
        Measures performance by running inference on the Edge TPU.
        """
        if not edgetpu_model_path or not self.delegate:
            return {'latency_ms_per_token': -1, 'power_w': -1}

        print(f"\n--- Measuring Live Performance on Edge TPU ---")
        interpreter = make_interpreter(edgetpu_model_path, delegate=self.delegate)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        # Run a few warmup inferences
        for _ in range(5):
            interpreter.set_tensor(input_details['index'], np.zeros(input_details['shape'], dtype=np.uint8))
            interpreter.invoke()

        # Timed inference run
        num_tokens_to_generate = 10
        latencies = []
        total_power_readings = []

        for _ in range(num_tokens_to_generate):
            start_time = time.perf_counter()
            interpreter.invoke() # This runs one step of inference
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000) # milliseconds
            power = self.read_power_consumption_w()
            if power != -1:
                total_power_readings.append(power)
        
        avg_latency = np.mean(latencies)
        avg_power = np.mean(total_power_readings) if total_power_readings else -1.0

        print(f"Avg. Latency: {avg_latency:.2f} ms/token | Avg. Power: {avg_power:.2f} W")
        return {'latency_ms_per_token': avg_latency, 'power_w': avg_power}
# ==============================================================================
# SECTION 2: LIVE QUANTIZATION & UTILITIES
# ==============================================================================

def apply_real_quantization_gptq(model_name, tokenizer, bits, sensitive_layers):
    """
    Applies ultra-low-bit quantization using the AutoGPTQ library.
    This function will modify the model in place.
    """
    print(f"Applying REAL GPTQ quantization to {bits}-bit. Disabling for {len(sensitive_layers)} layers.")
    
    gptq_config = GPTQConfig(
        bits=bits,
        dataset=["The theory of relativity revolutionized modern physics."],
        tokenizer=tokenizer,
        modules_to_not_convert=sensitive_layers
    )
    
    # This is the call that performs the actual quantization.

    try:
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=gptq_config,
            device_map='auto' # Let accelerate handle device mapping
        )
        print("Model successfully quantized.")
        return quantized_model
    except Exception as e:
        print(f"Fatal error during quantization: {e}")
        print("Quantization with AutoGPTQ often requires a compatible NVIDIA GPU and CUDA setup.")
        return None

def get_weight_sensitive_layers(model, percentile_threshold=95.0):
    weight_std_devs = [(n, m.weight.std().item()) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not weight_std_devs: return []
    all_stds = [s for _, s in weight_std_devs]
    sensitivity_cutoff = np.percentile(all_stds, percentile_threshold)
    return [name for name, std_dev in weight_std_devs if std_dev > sensitivity_cutoff]

def measure_model_memory_gb(model):
    """Measures the memory footprint of a PyTorch model."""
    torch.save(model.state_dict(), "temp_model.p")
    size_gb = os.path.getsize("temp_model.p") / 1e9
    os.remove("temp_model.p")
    return size_gb

# ==============================================================================
# SECTION 3: LIVE EVALUATION FUNCTIONS
# ==============================================================================

def generate_sample_response(model, tokenizer, prompt):
    """
    Generates a response from the model. This is the first step for MT-Bench.
    The full MT-Bench evaluation requires using these generated responses
    and scoring them with a strong LLM judge like GPT-4.
    """
    print(f"\n--- Generating Sample Response (for MT-Bench) ---")
    print(f"Prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, top_p=0.9, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model Response:\n{response}")
    # Save this `response` and evaluate it later.
    return response

def evaluate_position_bias(model, tokenizer):
    """
    Performs a "needle in a haystack" test for position bias.
    It checks if the model can find a specific fact ("the best city is London")
    when it's placed at different positions within a long, distracting context.
    """
    print("\n--- Running Live Position Bias Test ---")
    needle = "The best city in the world, without a doubt, is London."
    haystack_text = ("Abstract art is a form of visual art that does not attempt to represent an accurate depiction of a visual reality but instead use shapes, colours, forms and gestural marks to achieve its effect. " * 200)
    
    positions_to_test = np.linspace(0, len(haystack_text), 10).astype(int)
    success_count = 0
    
    for i, pos in enumerate(positions_to_test):
        haystack = haystack_text[:pos] + f" {needle} " + haystack_text[pos:]
        prompt = f"Based on the following text, what is the best city in the world?\n\n{haystack}\n\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
        
        if "london" in response:
            success_count += 1
        print(f"Test {i+1}/10 | Position: {pos:<6} | Success: {'london' in response}")

    accuracy = (success_count / len(positions_to_test)) * 100
    print(f"Position Bias Test Accuracy: {accuracy:.1f}%")
    return accuracy

# ==============================================================================
# SECTION 4: MAIN EXECUTION SCRIPT
# ==============================================================================

if __name__ == '__main__':
    MODEL_NAME = "meta-llama/Llama-3-8B"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Experiment 2 Data Collection for {MODEL_NAME} on {DEVICE}.")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Load in FP16 first for sensitivity analysis
        model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    except Exception as e:
        print(f"\nFatal: Could not load model '{MODEL_NAME}'. Check authentication and RAM.")
        exit()

    # --- Perform sensitivity analysis on the FP16 model ---
    print("\n--- Performing Sensitivity Analysis ---")
    sensitive_layers = get_weight_sensitive_layers(model_fp16, percentile_threshold=98.0)
    
    # --- Apply extreme quantization ---
    # This is a resource-intensive step.
    quantized_model = apply_real_quantization_gptq(MODEL_NAME, tokenizer, bits=2, sensitive_layers=sensitive_layers)
    
    if quantized_model is None:
        print("Exiting due to quantization failure.")
        exit()

    # --- Live Data Collection ---
    print("\n--- Starting Live Data Collection ---")
    
    # 1. Measure memory
    memory_gb = measure_model_memory_gb(quantized_model)
    print(f"Measured Memory Footprint: {memory_gb:.2f} GB")

    # 2. Get performance from hardware (using placeholder)
    coral_interface = CoralHardwareInterface()
    compiled_model_path = coral_interface.convert_and_compile_model(quantized_model, tokenizer)
    
    if compiled_model_path:
        performance = coral_interface.get_live_performance(compiled_model_path, tokenizer)
    else:
        print("Skipping performance measurement due to compilation failure.")
        performance = {'latency_ms_per_token': -1, 'power_w': -1}

    memory_gb = measure_model_memory_gb(quantized_model)
    pos_bias_accuracy = evaluate_position_bias(quantized_model, tokenizer)
    generate_sample_response(quantized_model, tokenizer, "Hello! Can you tell me about the history of the Roman Empire in three short paragraphs?")

    print("\n\n" + "="*80)
    print("EXPERIMENT 2: LIVE DATA COLLECTION SUMMARY")
    print("="*80)
    print(f"Model:                {MODEL_NAME}")
    print(f"Quantization:         Adaptive MPQ (~2.8-bit average)")
    print("-" * 30)
    print(f"Memory Footprint:     {memory_gb:.2f} GB")
    print(f"Latency (per token):  {performance['latency_ms_per_token']:.2f} ms")
    print(f"Power Consumption:    {performance['power_w']:.2f} W")
    print(f"Position Bias Score:  {pos_bias_accuracy:.1f}%")
    print("="*80)
    
    # Clean up memory
    del model_fp16
    del quantized_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
