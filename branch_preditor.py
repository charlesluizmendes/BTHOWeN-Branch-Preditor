import numpy as np
import sys
from collections import deque

# Funções auxiliares para manipulação de bits e XOR
def pc_binary(n, bit_size=32):
    return [(n >> i) & 1 for i in range(bit_size - 1, -1, -1)]

def xor_pc_ghr(pc_bits, ghr):
    return [pc_bit ^ ghr_bit for pc_bit, ghr_bit in zip(pc_bits, ghr)]

# Inicialização do histórico global (GHR) e local (LHR)
def initialize_local_history(lhr_size, lhr_bits_pc):
    return [[0] * lhr_size for _ in range(1 << lhr_bits_pc)]

def update_global_history(ghr, outcome):
    ghr.popleft()  # Remove o elemento mais antigo do GHR
    ghr.append(outcome)  # Adiciona o novo resultado de predição ao GHR

# Simulação do preditor BTHOWeN
class BTHOWeN:
    def __init__(self, address_size, input_size):
        self.history_table = {}
        self.address_size = address_size
        self.input_size = input_size

    def classify(self, input_data):
        # Converte o input_data para uma chave hashável para busca na tabela de histórico
        key = tuple(input_data)
        return self.history_table.get(key, 0)  # Retorna 0 se não encontrado

    def train(self, input_data, outcome):
        # Armazena o resultado correto para o input_data
        key = tuple(input_data)
        self.history_table[key] = outcome

# Função para processamento e "treino"
def train_predictor(file_name, address_size, params, interval=10000):
    num_branches, num_branches_predicted = 0, 0
    ghr = deque([0] * 24, maxlen=24)  # Histórico global com deque para manipulação eficiente
    lhr = initialize_local_history(24, 12)  # Inicialização do histórico local

    # Calcula o input_size ajustado para maior correspondência com o WiSARD
    input_size = (params[0] * 24) + (params[1] * 24) + (params[2] * 24) + \
                 (params[3] * 24) + (params[4] * 16) + (params[5] * 9) + \
                 (params[6] * 7) + (params[7] * 5)

    predictor = BTHOWeN(address_size=address_size, input_size=input_size)

    with open(file_name, "r") as stream:
        for line in stream:
            pc, outcome = map(int, line.strip().split())
            
            num_branches += 1
            pc_bits = pc_binary(pc, 32)
            pc_bits_lower_24 = pc_bits[-24:]
            xor_pc_ghr24 = xor_pc_ghr(pc_bits_lower_24, list(ghr))

            # Preparação dos dados de entrada (train_data)
            train_data = (pc_bits_lower_24 * params[0] +
                          list(ghr) * params[1] +
                          xor_pc_ghr24 * params[2])

            # Adiciona mais informações ao train_data para maior correspondência com o input_size
            train_data += (pc_bits_lower_24[:16] * params[3] +  # Substituindo para alcançar input_size
                           xor_pc_ghr24[:9] * params[4] +  # Segmento de 9 bits
                           xor_pc_ghr24[:7] * params[5] +  # Segmento de 7 bits
                           xor_pc_ghr24[:5] * params[6] +  # Segmento de 5 bits
                           xor_pc_ghr24[:8] * params[7])   # Segmento de 8 bits

            # Limita o train_data ao tamanho do input_size desejado
            train_data = train_data[:input_size]

            # Realiza a predição
            predicted = predictor.classify(train_data)
            if predicted == outcome:
                num_branches_predicted += 1

            # Treina o preditor armazenando o resultado correto para esta entrada
            predictor.train(train_data, outcome)

            # Atualização dos registros de históricos
            update_global_history(ghr, outcome)

            # Exibe precisão parcial a cada intervalo especificado
            if num_branches % interval == 0:
                partial_accuracy = (num_branches_predicted / num_branches) * 100
                print(f"Branch number: {num_branches}")
                print(f"----- Partial Accuracy: {partial_accuracy:.2f}%\n")

    # Resultados finais formatados
    final_accuracy = (num_branches_predicted / num_branches) * 100
    print(" ----- Results ------")
    print(f"Predicted  branches: {num_branches_predicted}")
    print(f"Not predicted branches: {num_branches - num_branches_predicted}")
    print(f"Accuracy: {final_accuracy:.6f}")
    print(f"\n------ Size of ntuple (address_size): {address_size} -----")
    print(f"\n------ Size of each input: {predictor.input_size} -----\n")

# Parâmetros e execução do modelo
if __name__ == "__main__":
    if len(sys.argv) != 12:
        print("Please, write the file/path_to_file correctly!")
        sys.exit(0)
    
    file_name = sys.argv[1]
    address_size = int(sys.argv[2])
    params = list(map(int, sys.argv[3:11]))  # Parâmetros para o preditor
    train_predictor(file_name, address_size, params)
