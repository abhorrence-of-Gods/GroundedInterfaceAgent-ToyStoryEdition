# MicroPython script for Edge Encoder Benchmark
#
# INSTRUCTIONS:
# 1. Flash this script to your MicroPython device (e.g., Raspberry Pi Pico W).
# 2. Download the MobileNetV2 TFLite model for microcontrollers:
#    https://www.tensorflow.org/lite/microcontrollers/get_started
#    (e.g., person_detect.tflite, rename to model.tflite)
# 3. Place `model.tflite` in the root of your device's filesystem.
# 4. Connect a camera module (if available) or run with dummy data.
# 5. Run from the REPL: `import edge_encoder_benchmark`
#
# This script will measure the inference time (latency) for encoding an image
# into a latent vector 'z', which is crucial for meeting the real-time
# requirements of the Toy-Story central brain.

try:
    import ulab.numpy as np
    import utime
    import gc
    # tflite_runtime is a custom firmware build.
    # If not available, we fall back to a dummy implementation.
    import tflite_runtime.interpreter as tflite
except ImportError:
    print("WARNING: tflite_runtime or ulab not found. Using dummy functions.")
    # Create dummy objects and functions for PC-based testing
    class DummyInterpreter:
        def __init__(self):
            self._tensor_shape = (1, 1, 1, 1000) # mobilenet output

        def allocate_tensors(self): pass
        def set_tensor(self, details, value): pass
        def invoke(self): utime.sleep_ms(25) # Simulate 25ms latency
        def get_tensor(self, details): return np.random.rand(1, 1000).astype(np.float32)
        def get_input_details(self): return [{'shape': [1, 96, 96, 1]}]
        def get_output_details(self): return [{'index': 0}]

    class Utime:
        def ticks_ms(self): return int(time.time() * 1000)
        def ticks_diff(self, a, b): return b - a
        def sleep_ms(self, ms): time.sleep(ms / 1000)

    import numpy as np
    import time
    tflite = type("tflite", (), {"Interpreter": DummyInterpreter})()
    utime = Utime()
    gc = type("gc", (), {"collect": lambda: None})()


# --- Configuration ---
MODEL_PATH = "model.tflite"
IMG_WIDTH = 96
IMG_HEIGHT = 96
NUM_SAMPLES = 50

# --- Helper Functions ---

def get_dummy_image():
    """Generates a random grayscale image."""
    return np.random.rand(IMG_HEIGHT, IMG_WIDTH, 1).astype(np.float32)

def benchmark_inference(interpreter, input_details, output_details):
    """Runs inference and measures latency."""
    latencies = []
    for i in range(NUM_SAMPLES):
        # 1. Get image (in a real scenario, from a camera)
        img = get_dummy_image()
        
        # 2. Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img.reshape(input_details[0]['shape']))
        
        # 3. Run inference and measure time
        start_time = utime.ticks_ms()
        interpreter.invoke()
        end_time = utime.ticks_ms()
        
        latency = utime.ticks_diff(end_time, start_time)
        latencies.append(latency)
        
        # Optional: Print progress
        if i % 10 == 0:
            print(f"Sample {i}/{NUM_SAMPLES}, Latency: {latency} ms")
            
        utime.sleep_ms(10) # Give some time for other processes
        
    return latencies

# --- Main Execution ---

def run_benchmark():
    """Main function to load model and run the benchmark."""
    print("--- Edge Encoder Benchmark ---")
    
    # 1. Load the TFLite model
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model '{MODEL_PATH}'. {e}")
        print("Please ensure the TFLite model is on your device's filesystem.")
        return

    # 2. Get model input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output details: {output_details[0]}")
    
    # 3. Collect garbage to get a clean measurement
    gc.collect()
    
    # 4. Run the benchmark
    print(f"\nRunning benchmark for {NUM_SAMPLES} samples...")
    latencies = benchmark_inference(interpreter, input_details, output_details)
    
    # 5. Calculate and print results
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)
    fps = 1000 / avg_latency
    
    gc.collect()
    
    print("\n--- Benchmark Results ---")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print(f"Max Latency: {max_latency} ms")
    print(f"Min Latency: {min_latency} ms")
    print("-------------------------")
    
    if fps < 25:
        print("WARNING: FPS is below the recommended 25-30 Hz for real-time operation.")
        print("Consider using a smaller model or more powerful hardware.")
    else:
        print("SUCCESS: Performance meets real-time requirements.")

# To run automatically when the script is imported
if __name__ == "__main__" or __name__ == "__main__":
    run_benchmark() 