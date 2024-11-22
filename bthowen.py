import os
import sys
import csv
import mmh3
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from datetime import datetime

# Implementação do Filtro de Bloom para o BTHOWeN
class BloomFilter:   
    def __init__(self, size: int, num_hashes: int):
        # Inicializa os parâmetros do filtro
        self.size = size
        self.num_hashes = num_hashes
        self.bits = np.zeros(size, dtype=np.int8)  # Array de bits do filtro
        self.weights = np.zeros(size, dtype=np.int8)  # Pesos associados aos bits

    # Gera índices de hash usando MurmurHash3
    def get_hash_indices(self, data: np.ndarray, seed: int) -> List[int]:       
        indices = []
        data_bytes = data.tobytes()

        for i in range(self.num_hashes):
            # Usa uma única função de hash com diferentes sementes
            index = mmh3.hash(data_bytes, seed + i) % self.size
            indices.append(index)

        return indices

    # Obtém o peso para os dados de entrada fornecidos
    def get_weight(self, data: np.ndarray, seed: int) -> int:        
        total_weight = 0
        indices = self.get_hash_indices(data, seed)

        for idx in indices:
            if self.bits[idx]:  # Só conta o peso se o bit estiver setado
                total_weight += self.weights[idx]

        return total_weight

    # Atualiza os pesos usando aprendizado one-shot
    def update(self, data: np.ndarray, seed: int, error: int):
        indices = self.get_hash_indices(data, seed)

        for idx in indices:
            self.bits[idx] = 1  # Seta o bit do filtro de Bloom
            # Atualiza o peso com clipping
            self.weights[idx] = np.clip(self.weights[idx] + error, -128, 127)

# Implementação completa do BTHOWeN
class BTHOWeN:  
    def __init__(self, input_params: List[int]):
        # Parâmetros do filtro de Bloom
        self.num_filters = 3
        self.num_hashes = 3
        self.filter_size = 2**14  # Tamanho padrão

        # Parâmetros adicionais
        self.ntuple_size = input_params[0]
        self.pc_times = input_params[1]
        self.ghr_times = input_params[2]
        self.pc_ghr_times = input_params[3]
        self.lhr1_times = input_params[4]
        self.lhr2_times = input_params[5]
        self.lhr3_times = input_params[6]
        self.lhr4_times = input_params[7]
        self.lhr5_times = input_params[8]
        self.ga_times = input_params[9]

        # Inicializa o registro de história global
        self.ghr_size = 24
        self.ghr = np.zeros(self.ghr_size, dtype=np.uint8)

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
            self.lhrs.append(np.zeros((lhr_size, length), dtype=np.uint8))

        # Inicializa o endereço global
        self.ga_lower = 8
        self.ga_branches = 8
        self.ga = np.zeros(self.ga_lower * self.ga_branches, dtype=np.uint8)

        # Inicializa os filtros de Bloom
        self.filters = [BloomFilter(self.filter_size, self.num_hashes) for _ in range(self.num_filters)]

        # Calcula o tamanho total da entrada
        self.input_size = (
            self.pc_times * 24 +
            self.ghr_times * self.ghr_size +
            self.pc_ghr_times * self.ghr_size +
            sum(self.lhr_configs[i][0] * input_params[i+4] for i in range(5)) +
            self.ga_times * len(self.ga)
        )       

    # Extrai características conforme especificado no artigo
    def extract_features(self, pc: int) -> np.ndarray:        
        # Extrai bits do PC
        pc_bits = np.array([int(b) for b in format(pc & ((1 << 24) - 1), '024b')], dtype=np.uint8)
        pc_bits_repeated = np.tile(pc_bits, self.pc_times)

        # GHR
        ghr_repeated = np.tile(self.ghr, self.ghr_times)

        # PC XOR GHR
        pc_ghr_xor = np.bitwise_xor(pc_bits[:self.ghr_size], self.ghr)
        pc_ghr_xor_repeated = np.tile(pc_ghr_xor, self.pc_ghr_times)

        # LHRs
        lhr_features = []
        lhr_times_list = [self.lhr1_times, self.lhr2_times, self.lhr3_times, self.lhr4_times, self.lhr5_times]
        for i, (length, bits_pc) in enumerate(self.lhr_configs):
            if lhr_times_list[i] > 0:
                index = int(''.join(map(str, pc_bits[-bits_pc:])), 2)
                lhr = self.lhrs[i][index]
                lhr_repeated = np.tile(lhr, lhr_times_list[i])
                lhr_features.append(lhr_repeated)
        lhr_features_combined = np.concatenate(lhr_features) if lhr_features else np.array([], dtype=np.uint8)

        # GA
        ga_repeated = np.tile(self.ga, self.ga_times) if self.ga_times > 0 else np.array([], dtype=np.uint8)

        # Combina todas as características
        features = np.concatenate([
            pc_bits_repeated,
            ghr_repeated,
            pc_ghr_xor_repeated,
            lhr_features_combined,
            ga_repeated
        ])

        return features

    # Atualiza todos os registros de histórico
    def _update_histories(self, pc: int, outcome: int):
        # Atualiza GHR
        self.ghr = np.roll(self.ghr, -1)
        self.ghr[-1] = outcome

        # Atualiza LHRs
        pc_bits = np.array([int(b) for b in format(pc & ((1 << 24) - 1), '024b')], dtype=np.uint8)
        for i, (length, bits_pc) in enumerate(self.lhr_configs):
            index = int(''.join(map(str, pc_bits[-bits_pc:])), 2)
            self.lhrs[i][index] = np.roll(self.lhrs[i][index], -1)
            self.lhrs[i][index][-1] = outcome

        # Atualiza GA
        new_bits = pc_bits[-self.ga_lower:]
        self.ga = np.roll(self.ga, -self.ga_lower)
        self.ga[-self.ga_lower:] = new_bits

    # Faz predição e realiza treinamento one-shot se necessário
    def predict_and_train(self, pc: int, outcome: int) -> bool:       
        features = self.extract_features(pc)

        # Obtém votos de todos os filtros de Bloom
        total_vote = 0
        for i, bloom_filter in enumerate(self.filters):
            total_vote += bloom_filter.get_weight(features, i)

        # Faz a predição
        prediction = total_vote >= 0
        correct = (prediction == bool(outcome))

        # Aprendizado one-shot: atualiza apenas em caso de erro de predição
        if not correct:
            error = 1 if outcome else -1
            for i, bloom_filter in enumerate(self.filters):
                bloom_filter.update(features, i, error)

        # Atualiza históricos
        self._update_histories(pc, outcome)

        return correct    

# Função principal do programa
def main():
    # Verifica argumentos da linha de comando
    if len(sys.argv) != 12:
        print("Please provide correct number of arguments")
        sys.exit(1)

    try:

        # Obtém arquivo de entrada e parâmetros
        input_file = sys.argv[1]
        parameters = list(map(int, sys.argv[2:]))

        # Inicializa o preditor
        predictor = BTHOWeN(parameters)
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
                    print(f"Branch number: {num_branches}")
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
        plt.savefig(f"{output_dir}/{timestamp}-BTHOWeN-accuracy.png")
        plt.show()

        # Resultados finais
        final_accuracy = (num_predicted / num_branches) * 100

        # Salva a acurácia em arquivo
        os.makedirs("Results_accuracy", exist_ok=True)

        with open(f"{output_dir}/{timestamp}-BTHOWeN-accuracy.csv", "w", newline='') as e:  # Abre arquivo de resultados em modo append
            writer = csv.writer(e)
            writer.writerow(["Number of Branches Processed", "Accuracy (%)"])  # Cabeçalho do arquivo
            writer.writerows(zip(branches_processed, accuracies))  # Dados do gráfico
            
        with open(f"Results_accuracy/{input_file_base}-accuracy.csv", 'a') as f:
            f.write(f"{final_accuracy:.4f} BTHOWeN\n")

        print("\n----- Results ------")
        print(f"Predicted branches: {num_predicted}")
        print(f"Not predicted branches: {num_branches - num_predicted}")
        print(f"Accuracy: {final_accuracy:.4f}")
        print(f"\n------ Size of ntuple (address_size): {parameters[0]}")
        print(f"------ Size of each input: {predictor.input_size}")

    except FileNotFoundError:  # Trata erro de arquivo não encontrado
        print("Can't open file")  # Mostra mensagem de erro
        sys.exit(1)  # Encerra programa com código de erro
    except Exception as e:  # Trata outros erros possíveis
        print(f"Error: {str(e)}")  # Mostra mensagem de erro detalhada
        sys.exit(1)  # Encerra programa com código de erro

# Ponto de entrada do programa
if __name__ == "__main__":
    main()
