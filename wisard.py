import sys
from typing import List, Tuple, Dict
import random
from dataclasses import dataclass
import numpy as np

class RAM:
    def __init__(self, indexes: List[int] = None):
        self.addresses = indexes if indexes else []
        self.positions: Dict[int, int] = {}
        self.index = 0
        
        if len(self.addresses) > 64:
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
            bin_val = input_data[addr]
            index += bin_val * p
            p *= 2
        return index

class Discriminator:
    def __init__(self, address_size: int, entry_size: int):
        self.entry_size = entry_size
        if address_size < 2 or entry_size < 2 or entry_size < address_size:
            raise Exception("Invalid address or entry size!")
        
        self.set_ram_shuffle(address_size)
    
    def classify(self, input_data: List[int]) -> List[int]:
        return [ram.get_vote(input_data) for ram in self.rams]
    
    def train(self, input_data: List[int]):
        for ram in self.rams:
            ram.train(input_data)
    
    def set_ram_shuffle(self, address_size: int):
        num_rams = self.entry_size // address_size
        remain = self.entry_size % address_size
        indexes_size = self.entry_size
        
        if remain > 0:
            num_rams += 1
            indexes_size += address_size - remain
        
        self.rams = []
        indexes = list(range(self.entry_size))
        random.shuffle(indexes)
        
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
            
            if not ambiguity or biggest <= 1:
                break
                
        return labels

class Wisard:
    def __init__(self, address_size: int, input_size: int):
        self.address_size = address_size
        self.discriminators = []
        self.make_discriminator(0, input_size)
        self.make_discriminator(1, input_size)
    
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
    
    def make_discriminator(self, label: int, entry_size: int):
        while len(self.discriminators) <= label:
            self.discriminators.append(None)
        self.discriminators[label] = Discriminator(self.address_size, entry_size)

def pc_to_binary(n: int, bits: int = 32) -> List[int]:
    return [(n >> i) & 1 for i in range(bits-1, -1, -1)]

def update_history(history: List[int], outcome: int, max_size: int):
    history.append(outcome)
    if len(history) > max_size:
        history.pop(0)

def main():
    if len(sys.argv) != 12:
        print("Please provide correct number of arguments!")
        sys.exit(1)
    
    input_file = sys.argv[1]
    parameters = [int(float(arg)) for arg in sys.argv[2:]]
    
    ntuple_size = parameters[0]
    pc_times = parameters[1]
    ghr_times = parameters[2]
    pc_xor_ghr_times = parameters[3]
    lhr_times = parameters[4:9]
    gas = parameters[9]
    
    # Initialize histories
    ghr = [0] * 24
    lhr_configs = [
        (24, 12), (16, 10), (9, 9), (7, 7), (5, 5)
    ]
    
    lhrs = []
    for length, bits in lhr_configs:
        lhr_size = 1 << bits
        lhrs.append([[0] * length for _ in range(lhr_size)])
    
    ga = [0] * (8 * 8)  # 8 branches x 8 bits
    
    input_size = (pc_times * 24 + ghr_times * 24 + 
                 pc_xor_ghr_times * 24 + 
                 sum(times * length for times, (length, _) in zip(lhr_times, lhr_configs)) +
                 gas * len(ga))
    
    predictor = Wisard(ntuple_size, input_size)
    print(f"Input size: {input_size}")
    
    num_branches = 0
    num_predicted = 0
    interval = 10000
    
    with open(input_file, 'r') as f:
        for line in f:
            pc, outcome = map(int, line.strip().split())
            num_branches += 1
            
            # Prepare input data
            pc_bits = pc_to_binary(pc)
            pc_bits_lower = pc_bits[-24:]
            xor_pc_ghr = [p ^ g for p, g in zip(pc_bits_lower, ghr)]
            
            # Build training data
            train_data = []
            
            # Add PC bits
            train_data.extend(pc_bits_lower * pc_times)
            
            # Add global histories
            train_data.extend(ghr * ghr_times)
            
            # Add XORed data
            train_data.extend(xor_pc_ghr * pc_xor_ghr_times)
            
            # Add local histories
            for lhr, (length, bits), times in zip(lhrs, lhr_configs, lhr_times):
                index = (pc & ((1 << bits) - 1))
                train_data.extend(lhr[index] * times)
            
            # Add global addresses
            train_data.extend(ga * gas)
            
            # Predict and train
            result = predictor.classify(train_data)
            if result == outcome:
                num_predicted += 1
            
            if num_branches % interval == 0:
                accuracy = (num_predicted / num_branches) * 100
                print(f"Branch number: {num_branches}")
                print(f"Partial Accuracy: {accuracy:.2f}%\n")
            
            # Train the predictor
            predictor.train(train_data, outcome)
            
            # Update histories
            update_history(ghr, outcome, 24)
            
            for lhr, (_, bits) in zip(lhrs, lhr_configs):
                index = (pc & ((1 << bits) - 1))
                update_history(lhr[index], outcome, lhr_configs[0][0])
            
            # Update global addresses
            ga.extend(pc_bits[-8:])
            ga = ga[8:]
    
    # Final results
    accuracy = (num_predicted / num_branches) * 100
    print("\n----- Results ------")
    print(f"Predicted branches: {num_predicted}")
    print(f"Not predicted branches: {num_branches - num_predicted}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\n------ Size of ntuple (address_size): {ntuple_size} -----")
    print(f"\n------ Size of each input: {input_size} -----")
    
    # Save results
    with open(f"{input_file}-accuracy.csv", 'a') as f:
        f.write(f"{accuracy:.4f}\n")

if __name__ == "__main__":
    main()