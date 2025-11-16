import numpy as np

def main():
    # Load your 0.npy file
    input_data = np.load('test_input.npy')

    # Save the raw data to a new binary file
    input_data.tofile('test_input.bin')

    print("Created test_input.bin")

    output_data = np.load('test_output.npy')
    output_data.tofile('test_output.bin')
    print("Created test_output.bin")

if __name__ == "__main__":
    main()