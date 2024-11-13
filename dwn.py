# Importando bibliotecas necessárias
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import sys
import time

# Implementação da Rede Neural com Pesos Dinâmicos
class DynamicWeightNetwork:
    def __init__(self, input_size: int, hidden_size: int = 32):
        # Inicializa os parâmetros da rede
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Inicializa os pesos com valores aleatórios pequenos
        self.weights = np.random.uniform(-0.1, 0.1, (hidden_size, input_size))
        self.output_weights = np.random.uniform(-0.1, 0.1, hidden_size)
        self.learning_rate = 0.01  # Taxa de aprendizado
        
    # Função de ativação sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Propagação direta (forward pass)
    def forward(self, input_data: List[int]) -> Tuple[np.ndarray, float]:
        # Converte entrada para array numpy
        x = np.array(input_data)
        # Camada oculta
        hidden = self.sigmoid(np.dot(self.weights, x))
        # Camada de saída
        output = self.sigmoid(np.dot(self.output_weights, hidden))
        return hidden, output
    
    # Treina a rede
    def train(self, input_data: List[int], target: int):
        # Propagação direta
        hidden, output = self.forward(input_data)
        
        # Calcula erros
        output_error = target - output
        hidden_error = output_error * self.output_weights
        
        # Atualiza pesos
        x = np.array(input_data)
        self.output_weights += self.learning_rate * output_error * hidden
        self.weights += self.learning_rate * np.outer(hidden_error, x)

# Preditor baseado em Rede Neural com Pesos Dinâmicos
class DWNPredictor:
    def __init__(self, input_params: List[int]):
        # Parâmetros de entrada
        self.ntuple_size = input_params[0]    # tamanho da n-tupla
        self.pc_times = input_params[1]       # número de vezes do PC
        self.ghr_times = input_params[2]      # número de vezes do GHR
        self.pc_ghr_times = input_params[3]   # número de vezes do PC-GHR
        self.lhr1_times = input_params[4]     # número de vezes do LHR1
        self.lhr2_times = input_params[5]     # número de vezes do LHR2
        self.lhr3_times = input_params[6]     # número de vezes do LHR3
        self.lhr4_times = input_params[7]     # número de vezes do LHR4
        self.lhr5_times = input_params[8]     # número de vezes do LHR5
        self.gas_times = input_params[9]      # número de vezes do GAS
        
        # Inicializa registros de histórico
        self.ghr = [0] * 24  # Registro de história global
        
        # Configurações dos registros de história local
        self.lhr_configs = [
            (24, 12),  # (comprimento, bits_pc) para LHR1
            (16, 10),  # LHR2
            (9, 9),    # LHR3
            (7, 7),    # LHR4
            (5, 5),    # LHR5
        ]
        
        # Inicializa LHRs
        self.lhrs = []
        for length, bits_pc in self.lhr_configs:
            lhr_size = 1 << bits_pc
            self.lhrs.append(np.zeros((lhr_size, length), dtype=int))
        
        # Inicializa endereço global
        self.ga_lower = 8
        self.ga_branches = 8
        self.ga = [0] * (self.ga_lower * self.ga_branches)
        
        # Calcula tamanho da entrada
        self.input_size = (
            self.pc_times * 24 +
            self.ghr_times * 24 +
            self.pc_ghr_times * 24 +
            sum(self.lhr_configs[i][0] * input_params[i+4] for i in range(5)) +
            self.gas_times * len(self.ga)
        )
        
        # Inicializa rede DWN com 32 neurônios ocultos
        self.hidden_size = 32
        self.network = DynamicWeightNetwork(self.input_size, self.hidden_size)
    
    # Converte PC para binário
    def pc_to_binary(self, pc: int) -> List[int]:
        return [(pc >> i) & 1 for i in range(31, -1, -1)]
    
    # Obtém bits menos significativos do PC
    def get_pc_lower(self, pc_bits: List[int], n: int) -> List[int]:
        return pc_bits[-n:]
    
    # Realiza XOR entre dois vetores
    def xor_vectors(self, v1: List[int], v2: List[int], n: int) -> List[int]:
        return [a ^ b for a, b in zip(v1[:n], v2[:n])]
    
    # Atualiza registro de história global
    def update_ghr(self, outcome: int):
        self.ghr.pop(0)
        self.ghr.append(outcome)
    
    # Atualiza registro de história local
    def update_lhr(self, pc_bits: List[int], outcome: int):
        for i, (length, bits_pc) in enumerate(self.lhr_configs):
            index = int(''.join(map(str, pc_bits[-bits_pc:])), 2)
            self.lhrs[i][index] = np.roll(self.lhrs[i][index], -1)
            self.lhrs[i][index][-1] = outcome
    
    # Atualiza endereço global
    def update_ga(self, pc_bits: List[int]):
        new_bits = pc_bits[-self.ga_lower:]
        self.ga = self.ga[self.ga_lower:] + new_bits
    
    # Prepara os dados de entrada
    def prepare_input(self, pc: int) -> List[int]:
        pc_bits = self.pc_to_binary(pc)
        pc_lower = self.get_pc_lower(pc_bits, 24)
        pc_ghr_xor = self.xor_vectors(pc_lower, self.ghr, 24)
        
        input_data = []
        
        # Adiciona bits do PC
        input_data.extend(pc_lower * self.pc_times)
        
        # Adiciona GHR
        input_data.extend(self.ghr * self.ghr_times)
        
        # Adiciona PC XOR GHR
        input_data.extend(pc_ghr_xor * self.pc_ghr_times)
        
        # Adiciona LHRs
        for i, (length, bits_pc) in enumerate(self.lhr_configs):
            index = int(''.join(map(str, pc_bits[-bits_pc:])), 2)
            times = [self.lhr1_times, self.lhr2_times, self.lhr3_times,
                    self.lhr4_times, self.lhr5_times][i]
            input_data.extend(list(self.lhrs[i][index]) * times)
        
        # Adiciona GA
        input_data.extend(self.ga * self.gas_times)
        
        return input_data
    
    # Realiza predição e treinamento
    def predict_and_train(self, pc: int, outcome: int) -> bool:
        input_data = self.prepare_input(pc)
        
        # Obtém predição
        _, prediction = self.network.forward(input_data)
        predicted_outcome = 1 if prediction > 0.5 else 0
        
        # Treina a rede
        self.network.train(input_data, outcome)
        
        # Atualiza históricos
        pc_bits = self.pc_to_binary(pc)
        self.update_ghr(outcome)
        self.update_lhr(pc_bits, outcome)
        self.update_ga(pc_bits)
        
        return predicted_outcome == outcome

# Função principal do programa
def main():
    # Verifica argumentos da linha de comando
    if len(sys.argv) != 12:
        print("Please provide correct arguments!")
        sys.exit(1)
    
    # Obtém arquivo de entrada e parâmetros
    input_file = sys.argv[1]
    parameters = list(map(int, sys.argv[2:]))
    
    # Inicializa o preditor
    predictor = DWNPredictor(parameters)
    print(f"Input size: {predictor.input_size}")
    
    # Contadores para estatísticas
    num_branches = 0
    num_correct = 0
    interval = 10000
    
    # Processa arquivo de entrada
    with open(input_file, 'r') as f:
        for line in f:
            pc, outcome = map(int, line.strip().split())
            num_branches += 1
            
            # Realiza predição e treinamento
            if predictor.predict_and_train(pc, outcome):
                num_correct += 1
            
            # Imprime resultados parciais
            if num_branches % interval == 0:
                accuracy = (num_correct / num_branches) * 100
                print(f"Branch number: {num_branches}")
                print(f"----- Partial Accuracy: {accuracy:.2f}\n")
    
    # Calcula e imprime resultados finais
    final_accuracy = (num_correct / num_branches) * 100
    print("\n----- Results ------")
    print(f"Predicted branches: {num_correct}")
    print(f"Not predicted branches: {num_branches - num_correct}")
    print(f"Accuracy: {final_accuracy:.2f}%")
    print(f"\n------ Hidden layer size: {predictor.hidden_size} -----")
    print(f"\n------ Size of each input: {predictor.input_size} -----")
    
    # Salva acurácia em arquivo
    with open(f"{input_file}-accuracy.csv", 'a') as f:
        f.write(f"{final_accuracy:.4f} DWN\n")

# Ponto de entrada do programa
if __name__ == "__main__":
    main()