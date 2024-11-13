import numpy as np
from typing import List
import sys

# Classe LUT (Look-Up Table) que implementa uma tabela de pesquisa usada para o processamento de dados.
class LUT:
    def __init__(self, n_inputs: int):
        # Inicializa a LUT com um número especificado de entradas.
        self.n_inputs = n_inputs
        # Cria a tabela de pesquisa com valores aleatórios entre -0.1 e 0.1.
        self.table = np.random.uniform(-0.1, 0.1, 2**n_inputs)
    
    # Método que realiza a consulta na tabela, retornando o valor associado ao endereço.
    def forward(self, address: int) -> float:
        return float(self.table[address])
    
    # Método para atualizar a tabela com base em um gradiente fornecido.
    def update(self, address: int, gradient: float):
        gradient = float(gradient)
        # Atualiza todos os valores na tabela, levando em consideração a distância de Hamming.
        for i in range(len(self.table)):
            hamming_dist = bin(i ^ address).count('1')
            self.table[i] += gradient / (hamming_dist + 1)

# Classe DWNLayer, que implementa uma camada de rede usando várias LUTs.
class DWNLayer:
    def __init__(self, input_size: int, n_luts: int, lut_inputs: int):
        # Inicializa a camada com um número especificado de LUTs e entradas.
        self.input_size = input_size
        self.n_luts = n_luts
        self.lut_inputs = lut_inputs
        self.luts = [LUT(lut_inputs) for _ in range(n_luts)]
        
        # Gera mapeamentos aleatórios de entradas para cada LUT.
        rng = np.random.default_rng()
        self.mapping = []
        for _ in range(n_luts):
            indices = []
            while len(indices) < lut_inputs:
                idx = rng.integers(input_size)
                if idx not in indices:
                    indices.append(idx)
            self.mapping.append(indices)
    
    # Método que realiza a propagação para frente da camada.
    def forward(self, x: np.ndarray) -> np.ndarray:
        outputs = np.zeros(self.n_luts)
        x_array = np.asarray(x)
        
        # Calcula os endereços para cada LUT e consulta o valor correspondente.
        for i in range(self.n_luts):
            address = 0
            for j, idx in enumerate(self.mapping[i]):
                if x_array[idx] > 0:
                    address |= (1 << j)
            outputs[i] = self.luts[i].forward(address)
        
        return outputs
    
    # Método para realizar o retropropagação da camada.
    def backward(self, x: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        dx = np.zeros_like(x)
        g = gradients / self.n_luts
        x_array = np.asarray(x)
        
        # Calcula o gradiente e atualiza os valores das LUTs.
        for i in range(self.n_luts):
            address = 0
            for j, idx in enumerate(self.mapping[i]):
                if x_array[idx] > 0:
                    address |= (1 << j)
            
            self.luts[i].update(address, g[i])
            
            for idx in self.mapping[i]:
                dx[idx] += g[i]
        
        return dx

# Classe BranchPredictor, que implementa um preditor de desvios baseado em uma rede neural sem peso (DWN).
class BranchPredictor:
    def __init__(self, address_size: int, input_params: List[int]):
        # Inicializa os parâmetros de configuração do preditor de desvios.
        self.pc_times = input_params[1]
        self.ghr_times = input_params[2]
        self.pc_ghr_times = input_params[3]
        self.lhr1_times = input_params[4]
        self.lhr2_times = input_params[5]
        self.lhr3_times = input_params[6]
        self.lhr4_times = input_params[7]
        self.lhr5_times = input_params[8]
        self.gas_times = input_params[9]

        # Inicializa os registradores de histórico global (GHR) e locais (LHR).
        self.ghr = np.zeros(24, dtype=np.int8)
        self.lhr_configs = [(24, 12), (16, 10), (9, 9), (7, 7), (5, 5)]
        self.lhr_times = [self.lhr1_times, self.lhr2_times, self.lhr3_times,
                         self.lhr4_times, self.lhr5_times]
        
        # Inicializa os registros de histórico local.
        self.lhrs = []
        for length, bits_pc in self.lhr_configs:
            lhr_size = 1 << bits_pc
            self.lhrs.append(np.zeros((lhr_size, length), dtype=np.int8))
        
        # Inicializa o histórico global de acessos.
        self.ga = np.zeros(64, dtype=np.int8)
        
        # Calcula o tamanho da entrada baseado nos parâmetros de configuração.
        self.input_size = (
            self.pc_times * 24 +
            self.ghr_times * 24 +
            self.pc_ghr_times * 24 +
            sum(length * times for (length, _), times in 
                zip(self.lhr_configs, self.lhr_times)) +
            self.gas_times * 64
        )
        
        # Inicializa três camadas DWN para a rede.
        self.layer1 = DWNLayer(self.input_size, 32, 6)
        self.layer2 = DWNLayer(32, 16, 6)
        self.layer3 = DWNLayer(16, 1, 6)
        
        self.pc_bits_cache = np.zeros(32, dtype=np.int8)
        self.input_buffer = np.zeros(self.input_size, dtype=np.int8)
        
        # Define a taxa de aprendizado para a retropropagação.
        self.learning_rate = 0.01
    
    # Converte o valor do PC (Program Counter) para uma representação binária.
    def pc_to_binary(self, pc: int):
        for i in range(32):
            self.pc_bits_cache[i] = (pc >> i) & 1
        return self.pc_bits_cache
    
    # Prepara a entrada da rede a partir do valor do PC.
    def prepare_input(self, pc: int) -> np.ndarray:
        pc_bits = self.pc_to_binary(pc)
        pc_lower = pc_bits[-24:]
        
        pos = 0
        buffer = self.input_buffer
        
        # Preenche o buffer de entrada com diferentes combinações dos dados.
        for _ in range(self.pc_times):
            buffer[pos:pos+24] = pc_lower
            pos += 24
        
        for _ in range(self.ghr_times):
            buffer[pos:pos+24] = self.ghr
            pos += 24
        
        pc_ghr_xor = pc_lower ^ self.ghr
        for _ in range(self.pc_ghr_times):
            buffer[pos:pos+24] = pc_ghr_xor
            pos += 24
        
        for i, ((length, bits_pc), times) in enumerate(zip(self.lhr_configs, self.lhr_times)):
            idx = int(''.join(map(str, pc_bits[-bits_pc:])), 2)
            lhr_data = self.lhrs[i][idx]
            for _ in range(times):
                buffer[pos:pos+length] = lhr_data
                pos += length
        
        for _ in range(self.gas_times):
            buffer[pos:pos+64] = self.ga
            pos += 64
        
        return buffer
    
    # Atualiza o histórico com base no resultado do desvio.
    def update_history(self, pc: int, outcome: int):
        pc_bits = self.pc_bits_cache
        
        # Atualiza o GHR com o resultado mais recente do desvio.
        self.ghr = np.roll(self.ghr, -1)
        self.ghr[-1] = outcome
        
        # Atualiza os LHRs com base no resultado mais recente do desvio.
        for i, (_, bits_pc) in enumerate(self.lhr_configs):
            idx = int(''.join(map(str, pc_bits[-bits_pc:])), 2)
            self.lhrs[i][idx] = np.roll(self.lhrs[i][idx], -1)
            self.lhrs[i][idx][-1] = outcome
        
        # Atualiza o histórico global de acessos.
        self.ga = np.roll(self.ga, -8)
        self.ga[-8:] = pc_bits[-8:]
    
    # Método que realiza a predição e o treinamento do preditor.
    def predict_and_train(self, pc: int, outcome: int) -> bool:
        x = self.prepare_input(pc)
        h1 = self.layer1.forward(x)
        h2 = self.layer2.forward(h1)
        prediction = self.layer3.forward(h2)[0]
        
        # Determina se o desvio foi tomado.
        is_taken = prediction > 0.5
        gradient = self.learning_rate * (float(outcome) - float(is_taken))
        
        # Realiza a retropropagação para ajustar os pesos.
        g3 = np.array([gradient])
        g2 = self.layer3.backward(h2, g3)
        g1 = self.layer2.backward(h1, g2)
        self.layer1.backward(x, g1)
        
        # Atualiza o histórico com base no resultado.
        self.update_history(pc, outcome)
        
        return is_taken == bool(outcome)

# Função principal que executa a lógica do preditor.
def main():
    if len(sys.argv) != 12:
        print("Please provide correct arguments!")
        sys.exit(1)
    
    # Lê o arquivo de entrada e os parâmetros fornecidos na linha de comando.
    input_file = sys.argv[1]
    parameters = list(map(int, sys.argv[2:]))
    
    predictor = BranchPredictor(parameters[0], parameters)
    print(f"Input size: {predictor.input_size}")
    
    num_branches = num_correct = 0
    interval = 10000
    
    # Abre o arquivo de entrada para leitura dos desvios.
    with open(input_file, 'r') as f:
        for line in f:
            pc, outcome = map(int, line.strip().split())
            num_branches += 1
            
            # Realiza a predição e atualiza a contagem de acertos.
            if predictor.predict_and_train(pc, outcome):
                num_correct += 1
            
            # Exibe a precisão parcial a cada 'interval' desvios.
            if num_branches % interval == 0:
                accuracy = (num_correct / num_branches) * 100
                print(f"Branch number: {num_branches}")
                print(f"----- Partial Accuracy: {accuracy:.2f}\n")
    
    # Exibe os resultados finais após processar todos os desvios.
    final_accuracy = (num_correct / num_branches) * 100
    print("\n----- Results ------")
    print(f"Predicted branches: {num_correct}")
    print(f"Not predicted branches: {num_branches - num_correct}")
    print(f"Accuracy: {final_accuracy:.2f}%")
    print(f"\n------ Size of ntuple (address_size): {parameters[0]} -----")
    print(f"\n------ Size of each input: {predictor.input_size} -----")
    
    # Escreve a precisão final em um arquivo CSV.
    with open(f"{input_file}-accuracy.csv", 'a') as f:
        f.write(f"{final_accuracy:.4f} DWN\n")

# Executa a função principal quando o script é chamado diretamente.
if __name__ == "__main__":
    main()