import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import sys
import time

class RAM:
    def __init__(self, indexes: List[int] = None):
        self.addresses = indexes if indexes else []
        self.positions: Dict[int, int] = {}
        self.index = 0
        
        if indexes and len(indexes) > 64:
            raise Exception("The base power to addressSize passed the limit of 2^64!")
    
    def get_vote(self, input_data: List[int]) -> int:
        self.index = self.get_index(input_data)
        return self.positions.get(self.index, 0)
    
    def train(self, input_data: List[int]):
        self.index = self.get_index(input_data)
        self.positions[self.index] = self.positions.get(self.index, 0) + 1
    
    def get_index(self, input_data: List[int]) -> int:
        index = 0
        p = 1
        for addr in self.addresses:
            bit = input_data[addr]
            index += bit * p
            p *= 2
        return index

class Discriminator:
    def __init__(self, address_size: int, entry_size: int):
        self.entry_size = entry_size
        self.rams: List[RAM] = []
        self.set_ram_shuffle(address_size)
    
    def classify(self, input_data: List[int]) -> List[int]:
        return [ram.get_vote(input_data) for ram in self.rams]
    
    def train(self, input_data: List[int]):
        for ram in self.rams:
            ram.train(input_data)
    
    def set_ram_shuffle(self, address_size: int):
        if address_size < 2:
            raise Exception("The address size cannot be less than 2!")
        if self.entry_size < 2:
            raise Exception("The entry size cannot be less than 2!")
        if self.entry_size < address_size:
            raise Exception("The address size cannot be bigger than entry size!")
        
        num_rams = self.entry_size // address_size
        remain = self.entry_size % address_size
        indexes_size = self.entry_size
        
        if remain > 0:
            num_rams += 1
            indexes_size += address_size - remain
        
        indexes = list(range(self.entry_size))
        np.random.shuffle(indexes)
        
        self.rams = []
        for i in range(num_rams):
            sub_indexes = indexes[i*address_size:(i+1)*address_size]
            self.rams.append(RAM(sub_indexes))

class Bleaching:
    @staticmethod
    def make(all_votes: List[List[int]]) -> List[int]:
        labels = [0, 0]
        bleaching = 1
        biggest = 0
        ambiguity = False
        
        while True:
            for i in range(2):
                labels[i] = sum(1 for vote in all_votes[i] if vote >= bleaching)
            
            bleaching += 1
            biggest = max(labels)
            ambiguity = labels.count(biggest) > 1
            
            if not (ambiguity and biggest > 1):
                break
                
        return labels

class WiSARD:
    def __init__(self, address_size: int, input_size: int):
        self.address_size = address_size
        self.discriminators = [
            Discriminator(address_size, input_size),
            Discriminator(address_size, input_size)
        ]
    
    def train(self, input_data: List[int], label: int):
        self.discriminators[label].train(input_data)
    
    def classify(self, input_data: List[int]) -> int:
        candidates = self.classify2(input_data)
        return 0 if candidates[0] >= candidates[1] else 1
    
    def classify2(self, input_data: List[int]) -> List[int]:
        all_votes = [
            self.discriminators[0].classify(input_data),
            self.discriminators[1].classify(input_data)
        ]
        return Bleaching.make(all_votes)

class BranchPredictor:
    def __init__(self, address_size: int, input_params: List[int]):
        self.ntuple_size = input_params[0]  # parameter1
        self.pc_times = input_params[1]     # parameter2
        self.ghr_times = input_params[2]    # parameter3
        self.pc_ghr_times = input_params[3] # parameter4
        self.lhr1_times = input_params[4]   # parameter5
        self.lhr2_times = input_params[5]   # parameter6
        self.lhr3_times = input_params[6]   # parameter7
        self.lhr4_times = input_params[7]   # parameter8
        self.lhr5_times = input_params[8]   # parameter9
        self.gas_times = input_params[9]    # parameter12
        
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
        
        # Initialize WiSARD
        self.wisard = WiSARD(address_size, self.input_size)
        
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
        prediction = self.wisard.classify(input_data)
        
        # Train after prediction
        self.wisard.train(input_data, outcome)
        
        # Update histories
        pc_bits = self.pc_to_binary(pc)
        self.update_ghr(outcome)
        self.update_lhr(pc_bits, outcome)
        self.update_ga(pc_bits)
        
        return prediction == outcome

def main():
    if len(sys.argv) != 12:
        print("Please provide correct arguments!")
        sys.exit(1)
    
    input_file = sys.argv[1]
    parameters = list(map(int, sys.argv[2:]))
    
    predictor = BranchPredictor(parameters[0], parameters)
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
    print(f"\n------ Size of ntuple (address_size): {parameters[0]} -----")
    print(f"\n------ Size of each input: {predictor.input_size} -----")
    
    # Save accuracy to file
    with open(f"{input_file}-accuracy.csv", 'a') as f:
        f.write(f"{final_accuracy:.4f} WISARD\n")

if __name__ == "__main__":
    main()