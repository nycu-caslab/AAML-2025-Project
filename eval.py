import argparse
import csv
import os
import re
import serial
from tqdm import tqdm # You must install this: pip install tqdm
import time
import numpy as np
import jiwer  # You must install this: pip install jiwer

# --- Configuration ---
# Model input shape (1, 296, 39)
INPUT_SIZE = 1 * 296 * 39  # 11544 bytes
# How many bytes of *data* to send per 'db' command (256 hex chars = 128 bytes)
SEND_BYTES_PER_CMD = 128

# Serial protocol
READY_MSG = 'm-ready\r\n'
# --- End Configuration ---

def parse_arg():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Wav2letter Final Grader Script")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", 
                        help="Serial port (e.g., /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=1843200, 
                        help="Baud rate (e.g., 1843200)")
    parser.add_argument("--csv", type=str, default="y_labels.csv", 
                        help="Path to the CSV file with labels.")
    return parser.parse_args()

class TestCase:
    """Holds data for a single test case from the CSV."""
    def __init__(self, filename, ground_truth) -> None:
        self.filename = filename
        self.ground_truth = str(ground_truth).lower().strip()

# Helper function to read and print messages
def read_and_print(com, until_msg):
    msg = com.read_until(until_msg.encode()).decode('utf-8', 'ignore')
    print(f"BOARD: {msg.strip()}")
    return msg
        
if __name__ == '__main__':
    args = parse_arg()
    
    # Load all test cases from y_labels.csv
    testcases = []
    try:
        with open(args.csv, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                # row[0] = filename, row[2] = ground_truth
                testcases.append(TestCase(row[0], row[2])) 
                print(f"Loaded test case: {row[0]} with ground truth: {row[2]}")
    except FileNotFoundError:
        print(f"Error: Cannot find CSV file: {args.csv}")
        exit()
        
    print(f"Loaded {len(testcases)} test cases from {args.csv}.")
    if len(testcases) == 0:
        print("Error: No test cases found in CSV.")
        exit()

    # --- Connect to Serial ---
    try:
        # Use a long timeout for 10-minute inference
        com = serial.Serial(args.port, args.baud, timeout=900) # 15-minute timeout
        print(f"Connected to {args.port} at {args.baud} baud.")
    except Exception as e:
        print(f"Error: Could not open serial port {args.port}. {e}")
        exit()
    if not com.is_open:
        print(f"Error: Serial port {args.port} is not open.")
        exit()

    print(f"Serial port {args.port} opened. Starting test...")
    # --- Run Evaluation ---
    results = {'total_latency_ms': 0.0}
    all_truth = []
    all_preds = []
    
    try:
        
        print("Sending '3' for Project Menu...")
        com.write(b'3\n')
        read_and_print(com, 'project> ')
        time.sleep(0.1) 

        print("Sending 'b' for Benchmark Mode...")
        com.write(b'b\n')
        read_and_print(com, READY_MSG) 
        print("Benchmark mode entered. Device is ready.")
        
        # --- Get Device Name ---
        com.write("name%".encode())
        read_and_print(com, READY_MSG)

        # --- Start the loop ---
        for testcase in tqdm(testcases, desc="Running Evaluation"):
            
            # 1. Load test data from file
            if not os.path.exists(testcase.filename):
                print(f"\nError: Input file not found: {testcase.filename}")
                raise FileNotFoundError# Skip to next test case
            input_data = np.fromfile(testcase.filename, dtype=np.int8)
                
            # if len(input_data) != INPUT_SIZE:
            #     print(f"\nSkipping {testcase.filename}: incorrect size.")
            #     continue
            with open(testcase.filename, 'rb') as test_input:
                com.read_all() # Clear any old data
                db_load_cmd = f"db load {INPUT_SIZE}%"
                print(f"SENDING: {db_load_cmd.strip()}")
                com.write(db_load_cmd.encode())
                read_and_print(com, READY_MSG)

                print(f"Sending {INPUT_SIZE} bytes of input data...")
                bytes_sent = 0
                data_chunk = test_input.read(SEND_BYTES_PER_CMD)
                while len(data_chunk) > 0:
                    db_data_cmd = f"db {data_chunk.hex()}%"
                    com.write(db_data_cmd.encode())
                    # Don't print the 361 "m-ready" messages
                    com.read_until(READY_MSG.encode()) 
                    bytes_sent += len(data_chunk)
                    data_chunk = test_input.read(SEND_BYTES_PER_CMD)
            print(f"Finished sending {bytes_sent} bytes.")

            infer_cmd = "infer 1 0%"
            print(f"SENDING: {infer_cmd.strip()}")
            com.write(infer_cmd.encode())
            
                
            # <<< MODIFIED: This is the new byte-by-byte loop to print dots >>>
            print(f"\n... Inference for '{testcase.filename}' running (will print dots):")
            msg_buffer_bytes = b''
            while True:
                byte_char = com.read(1)
                if not byte_char:
                    print("\nError: Read timed out during inference!")
                    msg = msg_buffer_bytes.decode('utf-8', 'ignore')
                    break
                try:
                    print(byte_char.decode('utf-8', 'ignore'), end='', flush=True)
                except:
                    pass
                msg_buffer_bytes += byte_char
                if msg_buffer_bytes.endswith(READY_MSG.encode()):
                    msg = msg_buffer_bytes.decode('utf-8', 'ignore')
                    break
            
            print("\n... Inference complete.")
            
            # Parse Latency
            try:
                m = [int(x) for x in re.findall(r'm-lap-us-([0-9]*)', msg)]
                latency_us = (m[1] - m[0]) 
                results['total_latency_ms'] += (latency_us / 1000.0) # Convert us to ms
            except Exception:
                print(f"Error parsing latency. Full message:\n{msg}")
                continue
                
            # Parse Text Result
            try:
                m = re.search(r'm-results-s\[(.*?)\]', msg)
                predicted_text = m.group(1).lower().strip()
            except Exception:
                print(f"Error parsing results string. Full message:\n{msg}")
                continue

            # 6. Store results for accuracy calculation
            all_truth.append(testcase.ground_truth)
            all_preds.append(predicted_text)
            
            print(f"  Truth: '{testcase.ground_truth}'")
            print(f"  Pred:  '{predicted_text}'")

    finally:
        com.close()

    # --- Print Final Report ---
    num_tests = len(all_truth)
    if num_tests > 0:
        avg_latency = results['total_latency_ms'] / num_tests
        
        # Calculate final Word Error Rate (WER)
        error_metrics = jiwer.compute_measures(all_truth, all_preds)
        wer = error_metrics['wer']

        print("\n--- Evaluation Complete ---")
        print(f"Total Inferences: {num_tests}")
        print(f"Average Latency:  {avg_latency:.2f} ms")
        print(f"Accuracy (WER):   {(1.0 - wer) * 100:.2f}%")
        print(f"(Word Error Rate: {wer * 100:.2f}%)")
    else:
        print("\n--- No tests were successfully run. ---")