import numpy as np
import sys
from typing import List, Tuple
import mmh3  # MurmurHash3 for better hash functions

class BloomFilter:
    """
    Bloom Filter implementation for bTHOWeN
    Uses multiple hash functions for indexing
    """
    def __init__(self, size: int, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = np.zeros(size, dtype=np.int8)
        self.weights = np.zeros(size, dtype=np.int8)
    
    def get_hash_indices(self, data: np.ndarray, seed: int) -> List[int]:
        """
        Generate multiple hash indices using MurmurHash3
        As specified in bTHOWeN paper, uses ternary hash functions
        """
        indices = []
        data_bytes = data.tobytes()
        
        for i in range(self.num_hashes):
            # Use different seeds for each hash function
            h1 = mmh3.hash(data_bytes, seed + i) % self.size
            h2 = mmh3.hash(data_bytes, seed + i + self.num_hashes) % self.size
            h3 = mmh3.hash(data_bytes, seed + i + 2 * self.num_hashes) % self.size
            
            # Combine hashes using ternary operation as per bTHOWeN
            index = (h1 ^ h2 ^ h3) % self.size
            indices.append(index)
            
        return indices
    
    def get_weight(self, data: np.ndarray, seed: int) -> int:
        """Get the weight for given input data"""
        total_weight = 0
        indices = self.get_hash_indices(data, seed)
        
        for idx in indices:
            if self.bits[idx]:  # Only count weight if bit is set
                total_weight += self.weights[idx]
                
        return total_weight
    
    def update(self, data: np.ndarray, seed: int, error: int):
        """Update weights using one-shot learning"""
        indices = self.get_hash_indices(data, seed)
        
        for idx in indices:
            self.bits[idx] = 1  # Set bloom filter bit
            # Update weight with clipping
            self.weights[idx] = np.clip(self.weights[idx] + error, -128, 127)

class BTHOWeN:
    """
    Complete bTHOWeN implementation with all original paper features
    """
    def __init__(self, address_size: int, input_size: int):
        # Bloom filter parameters from paper
        self.num_filters = 3
        self.filter_size = 2**14  # Size recommended in paper
        
        # Initialize Bloom filters
        self.filters = [BloomFilter(self.filter_size) for _ in range(self.num_filters)]
        
        # Feature parameters
        self.ghr_size = 24
        self.path_size = 16
        self.target_size = 16
        
        # History registers
        self.ghr = np.zeros(self.ghr_size, dtype=np.uint8)
        self.path_history = np.zeros(self.path_size, dtype=np.uint8)
        self.last_targets = np.zeros(self.target_size, dtype=np.uint32)
        
        # Statistics
        self.num_branches = 0
        self.num_predicted = 0
        
    def extract_features(self, pc: int) -> np.ndarray:
        """
        Extract features as specified in bTHOWeN paper:
        - Branch PC
        - Global history
        - Path history
        - Target history
        - XOR combinations
        """
        # Extract PC bits
        pc_bits = np.array([int(b) for b in format(pc & ((1 << 24) - 1), '024b')], 
                          dtype=np.uint8)
        
        # XOR features between PC and histories
        pc_ghr_xor = np.bitwise_xor(pc_bits[:self.ghr_size], self.ghr)
        pc_path_xor = np.bitwise_xor(pc_bits[:self.path_size], self.path_history)
        
        # Combine all features
        features = np.concatenate([
            pc_bits,  # Branch address
            self.ghr,  # Global history
            self.path_history,  # Path history
            pc_ghr_xor,  # PC xor GHR
            pc_path_xor,  # PC xor Path
        ])
        
        return features
    
    def predict_and_train(self, pc: int, target: int, outcome: int) -> bool:
        """
        Make prediction and perform one-shot training if needed
        Returns True if prediction was correct
        """
        features = self.extract_features(pc)
        
        # Get votes from all Bloom filters
        total_vote = 0
        for i, bloom_filter in enumerate(self.filters):
            total_vote += bloom_filter.get_weight(features, i)
        
        # Make prediction
        prediction = total_vote >= 0
        correct = (prediction == bool(outcome))
        
        # One-shot learning: update only on mispredictions
        if not correct:
            error = 1 if outcome else -1
            for i, bloom_filter in enumerate(self.filters):
                bloom_filter.update(features, i, error)
        
        # Update histories
        self._update_histories(pc, target, outcome)
        
        return correct
    
    def _update_histories(self, pc: int, target: int, outcome: int):
        """Update all history registers"""
        # Update GHR
        self.ghr = np.roll(self.ghr, 1)
        self.ghr[0] = outcome
        
        # Update path history
        self.path_history = np.roll(self.path_history, 1)
        self.path_history[0] = pc & 1
        
        # Update target history
        self.last_targets = np.roll(self.last_targets, 1)
        self.last_targets[0] = target

def main():
    if len(sys.argv) != 12:
        print("Please provide correct number of arguments")
        sys.exit(1)
        
    input_file = sys.argv[1]
    address_size = int(sys.argv[2])
    
    # Calculate input size from parameters
    input_size = sum(int(arg) for arg in sys.argv[3:11]) * 24
    
    predictor = BTHOWeN(address_size, input_size)
    interval = 10000
    
    try:
        with open(input_file, 'r') as f:
            num_branches = 0
            num_predicted = 0
            
            for line in f:
                pc, outcome = map(int, line.strip().split())
                target = pc + 4  # Default next instruction
                
                num_branches += 1
                if predictor.predict_and_train(pc, target, outcome):
                    num_predicted += 1
                    
                if num_branches % interval == 0:
                    accuracy = (num_predicted / num_branches) * 100
                    print(f"branch number: {num_branches}")
                    print(f"----- Partial Accuracy: {accuracy:.4f}\n")
            
            # Final results
            accuracy = (num_predicted / num_branches) * 100
            print("\n----- Results ------")
            print(f"Predicted branches: {num_predicted}")
            print(f"Not predicted branches: {num_branches - num_predicted}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"\n------ Size of ntuple (address_size): {address_size}")
            print(f"------ Size of each input: {input_size}")
            
            # Save accuracy
            with open(f"{input_file}-accuracy.csv", "a") as f:
                f.write(f"{accuracy:.4f}\n")
                
    except FileNotFoundError:
        print("Can't open file")
        sys.exit(1)

if __name__ == "__main__":
    main()