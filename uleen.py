import os
import sys  
import csv
import mmh3
import numpy as np 
import matplotlib.pyplot as plt
from typing import List 
from datetime import datetime

class BloomFilter:
    def __init__(self, size: int, num_hash: int):
        self.size = size
        self.num_hash = num_hash
        self.weights = np.random.uniform(-1, 1, size)
        self.binary = False
    
    def query(self, pattern: np.ndarray) -> float:
        indices = self._get_indices(pattern)
        if self.binary:
            return 1 if min(self.weights[indices]) > 0 else -1
        return np.tanh(np.mean(self.weights[indices]))
    
    def _get_indices(self, pattern: np.ndarray) -> List[int]:
        indices = []
        data_bytes = pattern.tobytes()
        for i in range(self.num_hash):
            hash_val = mmh3.hash(data_bytes, seed=i) 
            indices.append(hash_val % self.size)
        return indices
    
    def update(self, pattern: np.ndarray, gradient: float, learning_rate: float):
        if not self.binary:
            indices = self._get_indices(pattern)
            self.weights[indices] += learning_rate * gradient
            self.weights = np.clip(self.weights, -1, 1)

class SubModel:
    def __init__(self, input_size: int, table_size: int, num_filters: int = 4):
        self.input_size = input_size
        self.num_filters = num_filters
        self.filters = [BloomFilter(table_size, num_hash=3) for _ in range(self.num_filters)]
        self.bias = 0.0
    
    def predict(self, pattern: np.ndarray) -> float:
        total = sum(f.query(pattern) for f in self.filters) + self.bias
        return total

class ULEEN:   
    # Inicializa o preditor com parâmetros específicos
    def __init__(self, input_params: List[int]):
        # Define os parâmetros de entrada
        self.ntuple_size = input_params[0]  # tamanho da n-tupla
        self.pc_times = input_params[1]     # número de vezes do PC
        self.ghr_times = input_params[2]    # número de vezes do GHR
        self.pc_ghr_times = input_params[3] # número de vezes do PC-GHR
        self.lhr1_times = input_params[4]   # número de vezes do LHR1
        self.lhr2_times = input_params[5]   # número de vezes do LHR2
        self.lhr3_times = input_params[6]   # número de vezes do LHR3
        self.lhr4_times = input_params[7]   # número de vezes do LHR4
        self.lhr5_times = input_params[8]   # número de vezes do LHR5
        self.gas_times = input_params[9]    # número de vezes do GAS
        
        # Inicializa o registro de história global
        self.ghr = [0] * 24
        
        # Configurações dos registros de história local
        self.lhr_configs = [
            (24, 12),  # (comprimento, bits_pc) para LHR1
            (16, 10),  # LHR2
            (9, 9),    # LHR3
            (7, 7),    # LHR4
            (5, 5),    # LHR5
        ]
        
        # Inicializa os LHRs
        self.lhrs = []
        for length, bits_pc in self.lhr_configs:
            lhr_size = 1 << bits_pc
            self.lhrs.append(np.zeros((lhr_size, length), dtype=int))
            
        # Inicializa o endereço global
        self.ga_lower = 8
        self.ga_branches = 8
        self.ga = [0] * (self.ga_lower * self.ga_branches)
        
        self.input_size = (
            self.pc_times * 24 +
            self.ghr_times * 24 +
            self.pc_ghr_times * 24 +
            sum(self.lhr_configs[i][0] * input_params[i+4] for i in range(5)) +
            self.gas_times * len(self.ga)
        )
        
        self.table_size = 2**14
        self.num_submodels = 3
        self.num_hash = 3
        self.dropout_rate = 0.5
        self.learning_rate = 0.01
        self.threshold = 0.0
        
        self.submodels = [
            SubModel(self.input_size, self.table_size) 
            for _ in range(self.num_submodels)
        ]
        
        self.perceptron_weights = np.zeros(self.input_size)
        self.perceptron_threshold = 2.14 * np.sqrt(self.input_size)
        
        self.threshold_counter = 0
        self.threshold_window = 1000
        self.min_threshold = -1.0
        self.max_threshold = 1.0
        
        self.recent_predictions = []
        self.max_recent = 10000

    def _preprocess_input(self, pc: int) -> np.ndarray:
        features = []
        
        pc_bits = [(pc >> i) & 1 for i in range(31, -1, -1)]
        pc_lower = pc_bits[-24:]
        features.extend(pc_lower * self.pc_times)
        
        features.extend(list(self.ghr) * self.ghr_times)
        
        pc_ghr_xor = [a ^ b for a, b in zip(pc_lower, self.ghr)]
        features.extend(pc_ghr_xor * self.pc_ghr_times)
        
        for i, (length, bits_pc) in enumerate(self.lhr_configs):
            index = int(''.join(map(str, pc_bits[-bits_pc:])), 2)
            times = [self.lhr1_times, self.lhr2_times, self.lhr3_times,
                    self.lhr4_times, self.lhr5_times][i]
            features.extend(list(self.lhrs[i][index]) * times)
        
        features.extend(list(self.ga) * self.gas_times)
        
        return np.array(features, dtype=np.int32)
    
    def _update_histories(self, pc: int, outcome: int):
        pc_bits = [(pc >> i) & 1 for i in range(31, -1, -1)]
        
        self.ghr = np.roll(self.ghr, -1)
        self.ghr[-1] = outcome
        
        for i, (length, bits_pc) in enumerate(self.lhr_configs):
            index = int(''.join(map(str, pc_bits[-bits_pc:])), 2)
            self.lhrs[i][index] = np.roll(self.lhrs[i][index], -1)
            self.lhrs[i][index][-1] = outcome
        
        new_bits = pc_bits[-self.ga_lower:]
        self.ga = np.concatenate((self.ga[self.ga_lower:], new_bits))
    
    def _adjust_threshold(self):
        recent_accuracy = np.mean(self.recent_predictions)
        if recent_accuracy < 0.98:
            self.threshold *= 0.95
        else:
            self.threshold *= 1.05
        self.threshold = np.clip(self.threshold, self.min_threshold, self.max_threshold)

    def predict_and_train(self, pc: int, outcome: int) -> bool:
        pattern = self._preprocess_input(pc)
        
        active_models = np.random.random(self.num_submodels) >= self.dropout_rate
        predictions = []
        for i, model in enumerate(self.submodels):
            if active_models[i]:
                predictions.append(model.predict(pattern))
        
        perceptron_output = np.dot(self.perceptron_weights, pattern)
        use_perceptron = not predictions
        
        if use_perceptron:
            prediction = perceptron_output >= self.perceptron_threshold
            confidence = abs(perceptron_output) / self.perceptron_threshold
        else:
            total_vote = np.mean(predictions)
            prediction = total_vote >= self.threshold
            confidence = abs(total_vote)
        
        correct = (prediction == bool(outcome))
        
        if not correct or confidence < 0.5:
            target = 1 if outcome else -1
            
            if not use_perceptron:
                gradient = target - np.mean(predictions)
                for i, model in enumerate(self.submodels):
                    if active_models[i]:
                        for filter in model.filters:
                            filter.update(pattern, gradient, self.learning_rate)
                        model.bias += self.learning_rate * gradient
            
            if use_perceptron or not correct:
                perceptron_gradient = target - np.sign(perceptron_output)
                self.perceptron_weights += self.learning_rate * perceptron_gradient * pattern
        
        self.threshold_counter += 1
        if self.threshold_counter >= self.threshold_window:
            self._adjust_threshold()
            self.threshold_counter = 0
        
        self._update_histories(pc, outcome)
        
        self.recent_predictions.append(correct)
        if len(self.recent_predictions) > self.max_recent:
            self.recent_predictions.pop(0)
        
        return correct   

# Função principal do programa
def main():
    # Verifica argumentos da linha de comando
    if len(sys.argv) != 12:
        print("Usage: python uleen.py <trace_file> <ntuple_size> <pc_times> <ghr_times> <pc_ghr_times> <lhr1_times> <lhr2_times> <lhr3_times> <lhr4_times> <lhr5_times> <gas_times>")
        sys.exit(1)
    
    try:

        # Obtém arquivo de entrada e parâmetros
        input_file = sys.argv[1]
        parameters = [int(arg) for arg in sys.argv[2:]]
        
        # Inicializa o preditor
        predictor = ULEEN(parameters)
        print(f"Input size: {predictor.input_size}")

        # Inicializa contadores        
        num_branches = 0
        num_predicted = 0
        interval = 10000
        
        # Inicializa listas para armazenar dados para o gráfico
        branches_processed = []
        accuracies = []
        
        # Processa arquivo de entrada
        with open(input_file, 'r') as f:     
            for line in f:
                pc, outcome = map(int, line.strip().split())
                # Realiza predição e treinamento
                num_branches += 1
                if predictor.predict_and_train(pc, outcome):
                    num_predicted += 1
                # Imprime resultados parciais
                if num_branches % interval == 0:
                    accuracy = (num_predicted / num_branches) * 100
                    # Adiciona acurácias ao vetor
                    branches_processed.append(num_branches)
                    accuracies.append(accuracy)
                    print(f"branch number: {num_branches}")
                    print(f"----- Partial Accuracy: {accuracy:.4f}\n")
            
        # Remove o caminho do diretório e a extensão, deixando apenas o nome do arquivo
        input_file_base = os.path.splitext(os.path.basename(input_file))[0]
        # Cria o diretório caso ele não exista
        output_dir = f"Results_accuracy/{input_file_base}"
        os.makedirs(output_dir, exist_ok=True)
        # Obtém a data e hora atual para nomear o arquivo
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Cria o gráfico de acurácias
        plt.figure(figsize=(10, 6))
        plt.plot(branches_processed, accuracies, marker='o')
        plt.title("Accuracy Over Time")
        plt.xlabel("Number of Branches Processed")
        plt.ylabel("Accuracy (%)")
        plt.grid()
        plt.savefig(f"{output_dir}/{timestamp}-ULEEN-accuracy.png")
        plt.show()

        # Resultados finais
        final_accuracy = (num_predicted / num_branches) * 100

        # Salva a acurácia em arquivo
        os.makedirs("Results_accuracy", exist_ok=True)

        with open(f"{output_dir}/{timestamp}-ULEEN-accuracy.csv", "w", newline='') as e: # Abre arquivo de resultados em modo append
            writer = csv.writer(e)
            writer.writerow(["Number of Branches Processed", "Accuracy (%)"]) # Cabeçalho do arquivo
            writer.writerows(zip(branches_processed, accuracies)) # Dados do gráfico

        with open(f"Results_accuracy/{input_file_base}-accuracy.csv", "a") as f:
            f.write(f"{final_accuracy:.4f} ULEEN\n")
        
        print("\n----- Results ------")
        print(f"Predicted branches: {num_predicted}")
        print(f"Not predicted branches: {num_branches - num_predicted}")
        print(f"Accuracy: {final_accuracy:.4f}")
        print(f"------ Size of each input: {predictor.input_size}")

    except FileNotFoundError:  # Trata erro de arquivo não encontrado
        print("Can't open file")  # Mostra mensagem de erro
        sys.exit(1)  # Encerra programa com código de erro
    except Exception as e:  # Trata outros erros possíveis
        print(f"Error: {str(e)}")  # Mostra mensagem de erro detalhada
        sys.exit(1)  # Encerra programa com código de erro

if __name__ == "__main__":
    main()