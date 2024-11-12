import numpy as np
import sys
from typing import List
from collections import defaultdict

class UleenTable:
    def __init__(self, size: int):
        self.size = size
        self.weights = np.zeros(size, dtype=np.int8)
        self.max_weight = 1
        self.min_weight = -2
    
    def get_weight(self, index: int) -> int:
        return self.weights[index]
    
    def update(self, index: int, error: int):
        current = self.weights[index]
        if error > 0:
            if current < self.max_weight:
                self.weights[index] += 1
        else:
            if current > self.min_weight:
                self.weights[index] -= 1

class ULEEN:
    def __init__(self, address_size: int, input_size: int):
        self.num_tables = 4
        self.table_size = 2**14
        self.tables = [UleenTable(self.table_size) for _ in range(self.num_tables)]
        
        self.ghr_size = 32
        self.local_size = 16
        self.path_size = 16
        
        self.ghr = np.zeros(self.ghr_size, dtype=np.int32)
        self.local_history = defaultdict(lambda: np.zeros(self.local_size, dtype=np.int32))
        self.path_history = np.zeros(self.path_size, dtype=np.int32)
        
        self.threshold = 0
    
    def _fold_history(self, history: np.ndarray, bits: int) -> int:
        """Fold history into specified number of bits using XOR"""
        folded = 0
        segments = len(history) // bits
        if segments == 0:
            segments = 1
            
        for i in range(bits):
            segment_value = 0
            for j in range(segments):
                idx = i + j * bits
                if idx < len(history):
                    segment_value ^= history[idx]
            folded |= (segment_value & 1) << i
            
        return folded & ((1 << bits) - 1)
    
    def _get_table_index(self, pc: int, history: np.ndarray) -> int:
        """Generate index for weight table using PC and history"""
        history_hash = self._fold_history(history, 14)
        pc_hash = pc & ((1 << 14) - 1)  # Use lower 14 bits
        return (pc_hash ^ history_hash) & (self.table_size - 1)
    
    def _get_indices(self, pc: int) -> List[int]:
        indices = []
        
        # Index using PC and GHR
        idx1 = self._get_table_index(pc, self.ghr)
        indices.append(idx1)
        
        # Index using PC and Local History
        local_hist = self.local_history[pc]
        idx2 = self._get_table_index(pc, local_hist)
        indices.append(idx2)
        
        # Index using PC and Path History
        idx3 = self._get_table_index(pc, self.path_history)
        indices.append(idx3)
        
        # Index using PC and combined histories
        combined = np.bitwise_xor(self.ghr[:16], local_hist[:16])
        idx4 = self._get_table_index(pc, combined)
        indices.append(idx4)
        
        return indices
    
    def predict_and_train(self, pc: int, outcome: int) -> bool:
        indices = self._get_indices(pc)
        
        # Sum weights from all tables
        total_vote = 0
        for i, idx in enumerate(indices):
            total_vote += self.tables[i].get_weight(idx)
        
        # Make prediction
        prediction = total_vote >= self.threshold
        correct = (prediction == bool(outcome))
        
        # Update only on mispredictions
        if not correct:
            error = 1 if outcome else -1
            for i, idx in enumerate(indices):
                self.tables[i].update(idx, error)
        
        # Update histories
        self._update_histories(pc, outcome)
        
        return correct
    
    def _update_histories(self, pc: int, outcome: int):
        # Update global history
        self.ghr = np.roll(self.ghr, 1)
        self.ghr[0] = outcome
        
        # Update local history
        self.local_history[pc] = np.roll(self.local_history[pc], 1)
        self.local_history[pc][0] = outcome
        
        # Update path history
        self.path_history = np.roll(self.path_history, 1)
        self.path_history[0] = pc & 1

def main():
    if len(sys.argv) != 12:
        print("Please provide correct number of arguments")
        sys.exit(1)
        
    input_file = sys.argv[1]
    address_size = int(sys.argv[2])
    input_size = sum(int(arg) for arg in sys.argv[3:11]) * 24
    
    predictor = ULEEN(address_size, input_size)
    interval = 10000
    
    try:
        with open(input_file, 'r') as f:
            num_branches = 0
            num_predicted = 0
            
            for line in f:
                pc, outcome = map(int, line.strip().split())
                
                num_branches += 1
                if predictor.predict_and_train(pc, outcome):
                    num_predicted += 1
                    
                if num_branches % interval == 0:
                    accuracy = (num_predicted / num_branches) * 100
                    print(f"branch number: {num_branches}")
                    print(f"----- Partial Accuracy: {accuracy:.4f}\n")
            
            accuracy = (num_predicted / num_branches) * 100
            print("\n----- Results ------")
            print(f"Predicted branches: {num_predicted}")
            print(f"Not predicted branches: {num_branches - num_predicted}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"\n------ Size of ntuple (address_size): {address_size}")
            print(f"------ Size of each input: {input_size}")
            
            with open(f"{input_file}-accuracy.csv", "a") as f:
                f.write(f"{accuracy:.4f}\n")
                
    except FileNotFoundError:
        print("Can't open file")
        sys.exit(1)

if __name__ == "__main__":
    main()