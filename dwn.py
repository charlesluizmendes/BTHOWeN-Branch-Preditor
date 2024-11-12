import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import sys
import time

class DynamicWeightNetwork:
    def __init__(self, input_size: int, hidden_size: int = 32):  # Changed default to 32
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Initialize weights with small random values
        self.weights = np.random.uniform(-0.1, 0.1, (hidden_size, input_size))
        self.output_weights = np.random.uniform(-0.1, 0.1, hidden_size)
        self.learning_rate = 0.01
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, input_data: List[int]) -> Tuple[np.ndarray, float]:
        # Convert input to numpy array
        x = np.array(input_data)
        # Hidden layer
        hidden = self.sigmoid(np.dot(self.weights, x))
        # Output layer
        output = self.sigmoid(np.dot(self.output_weights, hidden))
        return hidden, output
    
    def train(self, input_data: List[int], target: int):
        # Forward pass
        hidden, output = self.forward(input_data)
        
        # Compute errors
        output_error = target - output
        hidden_error = output_error * self.output_weights
        
        # Update weights
        x = np.array(input_data)
        self.output_weights += self.learning_rate * output_error * hidden
        self.weights += self.learning_rate * np.outer(hidden_error, x)

class DWNPredictor:
    def __init__(self, input_params: List[int]):
        self.ntuple_size = input_params[0]    # parameter1
        self.pc_times = input_params[1]       # parameter2
        self.ghr_times = input_params[2]      # parameter3
        self.pc_ghr_times = input_params[3]   # parameter4
        self.lhr1_times = input_params[4]     # parameter5
        self.lhr2_times = input_params[5]     # parameter6
        self.lhr3_times = input_params[6]     # parameter7
        self.lhr4_times = input_params[7]     # parameter8
        self.lhr5_times = input_params[8]     # parameter9
        self.gas_times = input_params[9]      # parameter12
        
        # Initialize history registers
        self.ghr = [0] * 24
        
        # LHR configurations
        self.lhr_configs = [
            (24, 12),  # (length, bits_pc) for LHR1
            (16, 10),  # LHR2
            (9, 9),    # LHR3
            (7, 7),    # LHR4
            (5, 5),    # LHR5
        ]
        
        # Initialize LHRs
        self.lhrs = []
        for length, bits_pc in self.lhr_configs:
            lhr_size = 1 << bits_pc
            self.lhrs.append(np.zeros((lhr_size, length), dtype=int))
        
        # Initialize global address
        self.ga_lower = 8
        self.ga_branches = 8
        self.ga = [0] * (self.ga_lower * self.ga_branches)
        
        # Calculate input size
        self.input_size = (
            self.pc_times * 24 +
            self.ghr_times * 24 +
            self.pc_ghr_times * 24 +
            sum(self.lhr_configs[i][0] * input_params[i+4] for i in range(5)) +
            self.gas_times * len(self.ga)
        )
        
        # Initialize DWN network with 32 hidden neurons
        self.hidden_size = 32  # Changed to 32
        self.network = DynamicWeightNetwork(self.input_size, self.hidden_size)
    
    def pc_to_binary(self, pc: int) -> List[int]:
        return [(pc >> i) & 1 for i in range(31, -1, -1)]
    
    def get_pc_lower(self, pc_bits: List[int], n: int) -> List[int]:
        return pc_bits[-n:]
    
    def xor_vectors(self, v1: List[int], v2: List[int], n: int) -> List[int]:
        return [a ^ b for a, b in zip(v1[:n], v2[:n])]
    
    def update_ghr(self, outcome: int):
        self.ghr.pop(0)
        self.ghr.append(outcome)
    
    def update_lhr(self, pc_bits: List[int], outcome: int):
        for i, (length, bits_pc) in enumerate(self.lhr_configs):
            index = int(''.join(map(str, pc_bits[-bits_pc:])), 2)
            self.lhrs[i][index] = np.roll(self.lhrs[i][index], -1)
            self.lhrs[i][index][-1] = outcome
    
    def update_ga(self, pc_bits: List[int]):
        new_bits = pc_bits[-self.ga_lower:]
        self.ga = self.ga[self.ga_lower:] + new_bits
    
    def prepare_input(self, pc: int) -> List[int]:
        pc_bits = self.pc_to_binary(pc)
        pc_lower = self.get_pc_lower(pc_bits, 24)
        pc_ghr_xor = self.xor_vectors(pc_lower, self.ghr, 24)
        
        input_data = []
        
        # Add PC bits
        input_data.extend(pc_lower * self.pc_times)
        
        # Add GHR
        input_data.extend(self.ghr * self.ghr_times)
        
        # Add PC XOR GHR
        input_data.extend(pc_ghr_xor * self.pc_ghr_times)
        
        # Add LHRs
        for i, (length, bits_pc) in enumerate(self.lhr_configs):
            index = int(''.join(map(str, pc_bits[-bits_pc:])), 2)
            times = [self.lhr1_times, self.lhr2_times, self.lhr3_times,
                    self.lhr4_times, self.lhr5_times][i]
            input_data.extend(list(self.lhrs[i][index]) * times)
        
        # Add GA
        input_data.extend(self.ga * self.gas_times)
        
        return input_data
    
    def predict_and_train(self, pc: int, outcome: int) -> bool:
        input_data = self.prepare_input(pc)
        
        # Get prediction
        _, prediction = self.network.forward(input_data)
        predicted_outcome = 1 if prediction > 0.5 else 0
        
        # Train network
        self.network.train(input_data, outcome)
        
        # Update histories
        pc_bits = self.pc_to_binary(pc)
        self.update_ghr(outcome)
        self.update_lhr(pc_bits, outcome)
        self.update_ga(pc_bits)
        
        return predicted_outcome == outcome

def main():
    if len(sys.argv) != 12:
        print("Please provide correct arguments!")
        sys.exit(1)
    
    input_file = sys.argv[1]
    parameters = list(map(int, sys.argv[2:]))
    
    predictor = DWNPredictor(parameters)
    print(f"Input size: {predictor.input_size}")
    
    num_branches = 0
    num_correct = 0
    interval = 10000
    
    with open(input_file, 'r') as f:
        for line in f:
            pc, outcome = map(int, line.strip().split())
            num_branches += 1
            
            if predictor.predict_and_train(pc, outcome):
                num_correct += 1
            
            if num_branches % interval == 0:
                accuracy = (num_correct / num_branches) * 100
                print(f"Branch number: {num_branches}")
                print(f"----- Partial Accuracy: {accuracy:.2f}\n")
    
    final_accuracy = (num_correct / num_branches) * 100
    print("\n----- Results ------")
    print(f"Predicted branches: {num_correct}")
    print(f"Not predicted branches: {num_branches - num_correct}")
    print(f"Accuracy: {final_accuracy:.2f}%")
    print(f"\n------ Hidden layer size: {predictor.hidden_size} -----")
    print(f"\n------ Size of each input: {predictor.input_size} -----")
    
    # Save accuracy to file
    with open(f"{input_file}-accuracy.csv", 'a') as f:
        f.write(f"{final_accuracy:.4f} DWN\n")

if __name__ == "__main__":
    main()