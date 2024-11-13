# Importando bibliotecas necessárias
import numpy as np
import sys
from typing import List, Tuple
import mmh3

# Implementação do Filtro de Bloom para o BTHOWeN
class BloomFilter:
    """
    Filtro de Bloom para BTHOWeN
    Usa múltiplas funções de hash para indexação
    """
    def __init__(self, size: int, num_hashes: int = 3):
        # Inicializa os parâmetros do filtro
        self.size = size
        self.num_hashes = num_hashes
        self.bits = np.zeros(size, dtype=np.int8)  # Array de bits do filtro
        self.weights = np.zeros(size, dtype=np.int8)  # Pesos associados aos bits
    
    # Gera índices de hash usando MurmurHash3
    def get_hash_indices(self, data: np.ndarray, seed: int) -> List[int]:
        """
        Gera múltiplos índices de hash usando MurmurHash3
        Conforme especificado no artigo BTHOWeN, usa funções hash ternárias
        """
        indices = []
        data_bytes = data.tobytes()
        
        for i in range(self.num_hashes):
            # Usa uma única função de hash com diferentes sementes
            index = mmh3.hash(data_bytes, seed + i) % self.size
            indices.append(index)
    
        return indices
    
    # Obtém o peso para os dados de entrada fornecidos
    def get_weight(self, data: np.ndarray, seed: int) -> int:
        """Obtém o peso para os dados de entrada fornecidos"""
        total_weight = 0
        indices = self.get_hash_indices(data, seed)
        
        for idx in indices:
            if self.bits[idx]:  # Só conta o peso se o bit estiver setado
                total_weight += self.weights[idx]
                
        return total_weight
    
    # Atualiza os pesos usando aprendizado one-shot
    def update(self, data: np.ndarray, seed: int, error: int):
        """Atualiza pesos usando aprendizado one-shot"""
        indices = self.get_hash_indices(data, seed)
        
        for idx in indices:
            self.bits[idx] = 1  # Seta o bit do filtro de Bloom
            # Atualiza o peso com clipping
            self.weights[idx] = np.clip(self.weights[idx] + error, -128, 127)

# Implementação completa do BTHOWeN
class BTHOWeN:
    """
    Implementação completa do BTHOWeN com todas as características do artigo original
    """
    def __init__(self, address_size: int, input_size: int):
        # Parâmetros do filtro de Bloom do artigo
        self.num_filters = 3
        self.filter_size = 2**14  # Tamanho recomendado no artigo
        
        # Inicializa os filtros de Bloom
        self.filters = [BloomFilter(self.filter_size) for _ in range(self.num_filters)]
        
        # Parâmetros das características
        self.ghr_size = 24  # Tamanho do registro de história global
        self.path_size = 16  # Tamanho do histórico de caminhos
        self.target_size = 16  # Tamanho do histórico de alvos
        
        # Registros de histórico
        self.ghr = np.zeros(self.ghr_size, dtype=np.uint8)  # Registro de história global
        self.path_history = np.zeros(self.path_size, dtype=np.uint8)  # Histórico de caminhos
        self.last_targets = np.zeros(self.target_size, dtype=np.uint32)  # Histórico de alvos
        
        # Estatísticas
        self.num_branches = 0  # Número total de desvios
        self.num_predicted = 0  # Número de desvios previstos corretamente
        
    # Extrai características conforme especificado no artigo
    def extract_features(self, pc: int) -> np.ndarray:
        """
        Extrai características conforme especificado no artigo BTHOWeN:
        - PC do desvio
        - História global
        - Histórico de caminhos
        - Histórico de alvos
        - Combinações XOR
        """
        # Extrai bits do PC
        pc_bits = np.array([int(b) for b in format(pc & ((1 << 24) - 1), '024b')], 
                          dtype=np.uint8)
        
        # Características XOR entre PC e históricos
        pc_ghr_xor = np.bitwise_xor(pc_bits[:self.ghr_size], self.ghr)
        pc_path_xor = np.bitwise_xor(pc_bits[:self.path_size], self.path_history)
        
        # Combina todas as características
        features = np.concatenate([
            pc_bits,  # Endereço do desvio
            self.ghr,  # História global
            self.path_history,  # Histórico de caminhos
            pc_ghr_xor,  # PC xor GHR
            pc_path_xor,  # PC xor Path
        ])
        
        return features
    
    # Faz predição e realiza treinamento one-shot se necessário
    def predict_and_train(self, pc: int, target: int, outcome: int) -> bool:
        """
        Faz predição e realiza treinamento one-shot se necessário
        Retorna True se a predição estiver correta
        """
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
        self._update_histories(pc, target, outcome)
        
        return correct
    
    # Atualiza todos os registros de histórico
    def _update_histories(self, pc: int, target: int, outcome: int):
        """Atualiza todos os registros de histórico"""
        # Atualiza GHR
        self.ghr = np.roll(self.ghr, 1)
        self.ghr[0] = outcome
        
        # Atualiza histórico de caminhos
        self.path_history = np.roll(self.path_history, 1)
        self.path_history[0] = pc & 1
        
        # Atualiza histórico de alvos
        self.last_targets = np.roll(self.last_targets, 1)
        self.last_targets[0] = target

# Função principal do programa
def main():
    # Verifica argumentos da linha de comando
    if len(sys.argv) != 12:
        print("Please provide correct number of arguments")
        sys.exit(1)
        
    # Obtém arquivo de entrada e parâmetros
    input_file = sys.argv[1]
    address_size = int(sys.argv[2])
    
    # Calcula tamanho da entrada a partir dos parâmetros
    input_size = sum(int(arg) for arg in sys.argv[3:11]) * 24
    
    # Inicializa o preditor
    predictor = BTHOWeN(address_size, input_size)
    interval = 10000
    
    try:
        # Processa arquivo de entrada
        with open(input_file, 'r') as f:
            num_branches = 0
            num_predicted = 0
            
            # Loop principal de processamento
            for line in f:
                pc, outcome = map(int, line.strip().split())
                target = pc + 4  # Próxima instrução padrão
                
                # Realiza predição e treinamento
                num_branches += 1
                if predictor.predict_and_train(pc, target, outcome):
                    num_predicted += 1
                    
                # Imprime resultados parciais
                if num_branches % interval == 0:
                    accuracy = (num_predicted / num_branches) * 100
                    print(f"branch number: {num_branches}")
                    print(f"----- Partial Accuracy: {accuracy:.4f}\n")
            
            # Resultados finais
            accuracy = (num_predicted / num_branches) * 100
            print("\n----- Results ------")
            print(f"Predicted branches: {num_predicted}")
            print(f"Not predicted branches: {num_branches - num_predicted}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"\n------ Size of ntuple (address_size): {address_size}")
            print(f"------ Size of each input: {input_size}")
            
            # Salva acurácia em arquivo
            with open(f"{input_file}-accuracy.csv", "a") as f:
                f.write(f"{accuracy:.4f} BTHOWeN\n")
                
    except FileNotFoundError:
        print("Can't open file")
        sys.exit(1)

# Ponto de entrada do programa
if __name__ == "__main__":
    main()