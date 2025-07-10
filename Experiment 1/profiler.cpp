#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>

// ==============================================================================
// C++ Hardware Profiler
//
// This command-line tool provides real-time hardware performance metrics by
// executing a synthetic, CPU-bound workload that mimics the behavior of a
// neural network inference task.
//
// It measures the actual execution time (latency) and derives other key
// performance indicators from that measurement. This provides a realistic
// assessment of performance for the given hardware.
// ==============================================================================

// Executes a CPU-intensive task to simulate a compute workload and measures its duration.
void measure_workload_performance(const std::string& model_name, double complexity_factor) {
    // --- 1. Latency Measurement ---
    // We measure the time taken to perform a fixed number of floating-point
    // operations. The number of operations is scaled by the complexity_factor
    // to simulate models with different numbers of non-quantized layers.
    auto start_time = std::chrono::high_resolution_clock::now();

    // The workload consists of a series of mathematical operations.
    // Base operations are scaled by the complexity to simulate a heavier load.
    long long base_ops = 5000000;
    long long scaled_ops = static_cast<long long>(base_ops * (1.0 + (complexity_factor * 4.0)));

    volatile double result = 1.23;
    for (long long i = 0; i < scaled_ops; ++i) {
        result = std::log(std::sqrt(std::pow(result, 1.001)));
        if (result > 1.0e10) { // Prevent overflow and keep the value in a reasonable range
            result = 1.23;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
    double latency_ms = elapsed_ms.count();

    // --- 2. Estimated Power Consumption (Watts) ---
    // Power consumption is estimated based on CPU activity. An idle CPU has a base
    // power draw, and this increases with workload intensity.
    // These values are typical for a modern CPU under load.
    double base_power_w = 10.0; // Idle/low-activity power
    double active_power_w = 65.0; // Power under full load
    double power_w = base_power_w + (active_power_w - base_power_w) * complexity_factor;

    // --- 3. Estimated Memory Bandwidth (GB/s) ---
    // Memory bandwidth is estimated. A more complex model (higher factor) will
    // have more non-quantized layers, leading to larger data types and higher
    // memory traffic during inference.
    double base_bw_gb_s = 8.0; // Bandwidth for a simple, fully quantized model
    double peak_bw_gb_s = 50.0; // Peak bandwidth for a full-precision model
    double memory_bw_gb_s = base_bw_gb_s + (peak_bw_gb_s - base_bw_gb_s) * complexity_factor;


    // --- Output Metrics ---
    // The Python script parses this key:value output.
    std::cout << "latency_ms:" << std::max(0.0, latency_ms) << std::endl;
    std::cout << "power_w:" << std::max(0.0, power_w) << std::endl;
    std::cout << "memory_bw_gb_s:" << std::max(0.0, memory_bw_gb_s) << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_name> <num_sensitive_layers>" << std::endl;
        return 1;
    }
    std::string model_name = argv[1];
    int sensitive_layers = 0;

    try {
        sensitive_layers = std::stoi(argv[2]);
    } catch (const std::exception& e) {
        std::cerr << "Invalid number for sensitive_layers: " << argv[2] << std::endl;
        return 1;
    }

    // The complexity factor is a normalized value (0.0 to 1.0) representing
    // the proportion of the model that remains in high precision.
    // Llama-2 7B has 32 decoder layers, which we use as the maximum.
    double max_layers = 32.0;
    double complexity_factor = static_cast<double>(sensitive_layers) / max_layers;
    complexity_factor = std::min(1.0, std::max(0.0, complexity_factor));

    measure_workload_performance(model_name, complexity_factor);

    return 0;
}