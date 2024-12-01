import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from datetime import datetime

# Classe para implementar a memória RAM do WiSARD
class RAM:
    # Inicializa a RAM com endereços específicos
    def __init__(self, indexes: List[int] = None):
        self.addresses = indexes if indexes else []
        self.positions: Dict[int, int] = {}
        self.index = 0
        
        # Verifica se o número de endereços não excede o limite
        if indexes and len(indexes) > 64:
            raise Exception("The base power to addressSize passed the limit of 2^64!")
    
    # Obtém o voto baseado nos dados de entrada
    def get_vote(self, input_data: List[int]) -> int:
        self.index = self.get_index(input_data)
        return self.positions.get(self.index, 0)
    
    # Treina a RAM com os dados de entrada
    def train(self, input_data: List[int]):
        self.index = self.get_index(input_data)
        self.positions[self.index] = self.positions.get(self.index, 0) + 1
    
    # Calcula o índice baseado nos dados de entrada
    def get_index(self, input_data: List[int]) -> int:
        index = 0
        p = 1
        for addr in self.addresses:
            bit = input_data[addr]
            index += bit * p
            p *= 2
        return index

# Classe que implementa o discriminador do WiSARD
class Discriminator:
    # Inicializa o discriminador com tamanhos específicos
    def __init__(self, address_size: int, entry_size: int):
        self.entry_size = entry_size
        self.rams: List[RAM] = []
        self.set_ram_shuffle(address_size)
    
    # Classifica os dados de entrada usando as RAMs
    def classify(self, input_data: List[int]) -> List[int]:
        return [ram.get_vote(input_data) for ram in self.rams]
    
    # Treina o discriminador com os dados de entrada
    def train(self, input_data: List[int]):
        for ram in self.rams:
            ram.train(input_data)
    
    # Configura as RAMs com embaralhamento de endereços
    def set_ram_shuffle(self, address_size: int):
        # Verifica os tamanhos mínimos necessários
        if address_size < 2:
            raise Exception("The address size cannot be less than 2!")
        if self.entry_size < 2:
            raise Exception("The entry size cannot be less than 2!")
        if self.entry_size < address_size:
            raise Exception("The address size cannot be bigger than entry size!")
        
        # Calcula o número de RAMs necessárias
        num_rams = self.entry_size // address_size
        remain = self.entry_size % address_size
        indexes_size = self.entry_size
        
        # Ajusta o tamanho se necessário
        if remain > 0:
            num_rams += 1
            indexes_size += address_size - remain
        
        # Embaralha os índices
        indexes = list(range(self.entry_size))
        np.random.shuffle(indexes)
        
        # Cria as RAMs com os índices embaralhados
        self.rams = []
        for i in range(num_rams):
            sub_indexes = indexes[i*address_size:(i+1)*address_size]
            self.rams.append(RAM(sub_indexes))

# Classe que implementa o processo de bleaching (branqueamento)
class Bleaching:
    # Método estático para realizar o branqueamento dos votos
    @staticmethod
    def make(all_votes: List[List[int]]) -> List[int]:
        labels = [0, 0]
        bleaching = 1
        biggest = 0
        ambiguity = False
        
        # Loop até resolver ambiguidade
        while True:
            for i in range(2):
                labels[i] = sum(1 for vote in all_votes[i] if vote >= bleaching)
            
            bleaching += 1
            biggest = max(labels)
            ambiguity = labels.count(biggest) > 1
            
            if not (ambiguity and biggest > 1):
                break
                
        return labels

# Classe principal do WiSARD
class WiSARD:
    # Inicializa o WiSARD com tamanhos específicos
    def __init__(self, address_size: int, input_size: int):
        self.address_size = address_size
        self.discriminators = [
            Discriminator(address_size, input_size),
            Discriminator(address_size, input_size)
        ]
    
    # Treina o WiSARD com dados e rótulo
    def train(self, input_data: List[int], label: int):
        self.discriminators[label].train(input_data)
    
    # Classifica os dados de entrada
    def classify(self, input_data: List[int]) -> int:
        candidates = self.classify2(input_data)
        return 0 if candidates[0] >= candidates[1] else 1
    
    # Retorna os votos de classificação
    def classify2(self, input_data: List[int]) -> List[int]:
        all_votes = [
            self.discriminators[0].classify(input_data),
            self.discriminators[1].classify(input_data)
        ]
        return Bleaching.make(all_votes)

# Classe do preditor de desvios
class BranchPredictor:
    # Inicializa o preditor com parâmetros específicos
    def __init__(self, address_size: int, input_params: List[int]):
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
        
        # Calcula o tamanho total da entrada
        self.input_size = (
            self.pc_times * 24 +
            self.ghr_times * 24 +
            self.pc_ghr_times * 24 +
            sum(self.lhr_configs[i][0] * input_params[i+4] for i in range(5)) +
            self.gas_times * len(self.ga)
        )
        
        # Inicializa o WiSARD
        self.wisard = WiSARD(address_size, self.input_size)
        
    # Converte PC para binário
    def pc_to_binary(self, pc: int) -> List[int]:
        return [(pc >> i) & 1 for i in range(31, -1, -1)]
    
    # Obtém os bits menos significativos do PC
    def get_pc_lower(self, pc_bits: List[int], n: int) -> List[int]:
        return pc_bits[-n:]
    
    # Realiza XOR entre dois vetores
    def xor_vectors(self, v1: List[int], v2: List[int], n: int) -> List[int]:
        return [a ^ b for a, b in zip(v1[:n], v2[:n])]
    
    # Atualiza o registro de história global
    def update_ghr(self, outcome: int):
        self.ghr.pop(0)
        self.ghr.append(outcome)
    
    # Atualiza o registro de história local
    def update_lhr(self, pc_bits: List[int], outcome: int):
        for i, (length, bits_pc) in enumerate(self.lhr_configs):
            index = int(''.join(map(str, pc_bits[-bits_pc:])), 2)
            self.lhrs[i][index] = np.roll(self.lhrs[i][index], -1)
            self.lhrs[i][index][-1] = outcome
    
    # Atualiza o endereço global
    def update_ga(self, pc_bits: List[int]):
        new_bits = pc_bits[-self.ga_lower:]
        self.ga = self.ga[self.ga_lower:] + new_bits
    
    # Prepara os dados de entrada para o preditor
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
        prediction = self.wisard.classify(input_data)
        
        # Treina após a predição
        self.wisard.train(input_data, outcome)
        
        # Atualiza os históricos
        pc_bits = self.pc_to_binary(pc)
        self.update_ghr(outcome)
        self.update_lhr(pc_bits, outcome)
        self.update_ga(pc_bits)
        
        return prediction == outcome

# Função principal
def main():
    # Verifica argumentos de linha de comando
    if len(sys.argv) != 12:
        print("Please provide correct arguments!")
        sys.exit(1)
    
    try:

        # Obtém arquivo de entrada e parâmetros
        input_file = sys.argv[1]
        parameters = list(map(int, sys.argv[2:]))

        # Inicializa o preditor
        predictor = BranchPredictor(parameters[0], parameters)
        print(f"Input size: {predictor.input_size}")

        # Inicializa contadores
        num_branches = 0
        num_correct = 0
        interval = 10000

        # Inicializa listas para armazenar dados para o gráfico
        branches_processed = []
        accuracies = []

        # Processa o arquivo de entrada
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

                    # Adiciona acuracias ao vetor
                    branches_processed.append(num_branches) 
                    accuracies.append(accuracy) 

                    print(f"Branch number: {num_branches}")
                    print(f"----- Partial Accuracy: {accuracy:.2f}\n")
    
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
        plt.savefig(f"{output_dir}/{timestamp}-WISARD-accuracy.png")
        plt.show() 

        # Calcula e imprime resultados finais
        final_accuracy = (num_correct / num_branches) * 100

        # Salva a acurácia em arquivo
        os.makedirs("Results_accuracy", exist_ok=True)

        with open(f"{output_dir}/{timestamp}-WISARD-accuracy.csv", "w", newline='') as e:  # Abre arquivo de resultados em modo append
            writer = csv.writer(e)
            writer.writerow(["Number of Branches Processed", "Accuracy (%)"])  # Cabeçalho do arquivo
            writer.writerows(zip(branches_processed, accuracies))  # Dados do gráfico

        with open(f"Results_accuracy/{input_file_base}-accuracy.csv", 'a') as f:
            f.write(f"{final_accuracy:.4f} WISARD\n")
    
        print("\n----- Results ------")
        print(f"Predicted branches: {num_correct}")
        print(f"Not predicted branches: {num_branches - num_correct}")
        print(f"Accuracy: {final_accuracy:.2f}%")
        print(f"\n------ Size of ntuple (address_size): {parameters[0]} -----")
        print(f"\n------ Size of each input: {predictor.input_size} -----")

    except FileNotFoundError:  # Trata erro de arquivo não encontrado
        print("Can't open file")  # Mostra mensagem de erro
        sys.exit(1)  # Encerra programa com código de erro
    except Exception as e:  # Trata outros erros possíveis
        print(f"Error: {str(e)}")  # Mostra mensagem de erro detalhada
        sys.exit(1)  # Encerra programa com código de erro

# Ponto de entrada do programa
if __name__ == "__main__":
    main()