# Importando bibliotecas necessárias
import numpy as np
import sys
from typing import List
from collections import defaultdict

# Implementação da tabela de pesos do ULEEN
class UleenTable:
    def __init__(self, size: int):
        # Inicializa a tabela com tamanho especificado
        self.size = size
        self.weights = np.zeros(size, dtype=np.int8)  # Array de pesos
        self.max_weight = 1    # Peso máximo permitido
        self.min_weight = -2   # Peso mínimo permitido
    
    # Obtém o peso para um índice específico
    def get_weight(self, index: int) -> int:
        return self.weights[index]
    
    # Atualiza o peso com saturação nos limites
    def update(self, index: int, error: int):
        current = self.weights[index]
        if error > 0:  # Caso erro positivo, incrementa se não atingiu máximo
            if current < self.max_weight:
                self.weights[index] += 1
        else:  # Caso erro negativo, decrementa se não atingiu mínimo
            if current > self.min_weight:
                self.weights[index] -= 1

# Implementação do preditor ULEEN
class ULEEN:
    def __init__(self, address_size: int, input_size: int):
        # Configuração das tabelas de pesos
        self.num_tables = 4
        self.table_size = 2**14
        self.tables = [UleenTable(self.table_size) for _ in range(self.num_tables)]
        
        # Tamanhos dos registros de histórico
        self.ghr_size = 32     # Tamanho do histórico global
        self.local_size = 16   # Tamanho do histórico local
        self.path_size = 16    # Tamanho do histórico de caminho
        
        # Inicialização dos registros de histórico
        self.ghr = np.zeros(self.ghr_size, dtype=np.int32)  # Histórico global
        self.local_history = defaultdict(lambda: np.zeros(self.local_size, dtype=np.int32))  # Históricos locais
        self.path_history = np.zeros(self.path_size, dtype=np.int32)  # Histórico de caminho
        
        self.threshold = 0  # Limiar para decisão
    
    # Dobra o histórico em número específico de bits usando XOR
    def _fold_history(self, history: np.ndarray, bits: int) -> int:
        """Dobra o histórico em número específico de bits usando XOR"""
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
    
    # Gera índice para tabela de pesos usando PC e histórico
    def _get_table_index(self, pc: int, history: np.ndarray) -> int:
        """Gera índice para tabela de pesos usando PC e histórico"""
        history_hash = self._fold_history(history, 14)
        pc_hash = pc & ((1 << 14) - 1)  # Usa os 14 bits menos significativos
        return (pc_hash ^ history_hash) & (self.table_size - 1)
    
    # Obtém índices para todas as tabelas
    def _get_indices(self, pc: int) -> List[int]:
        indices = []
        
        # Índice usando PC e histórico global
        idx1 = self._get_table_index(pc, self.ghr)
        indices.append(idx1)
        
        # Índice usando PC e histórico local
        local_hist = self.local_history[pc]
        idx2 = self._get_table_index(pc, local_hist)
        indices.append(idx2)
        
        # Índice usando PC e histórico de caminho
        idx3 = self._get_table_index(pc, self.path_history)
        indices.append(idx3)
        
        # Índice usando PC e históricos combinados
        combined = np.bitwise_xor(self.ghr[:16], local_hist[:16])
        idx4 = self._get_table_index(pc, combined)
        indices.append(idx4)
        
        return indices
    
    # Realiza predição e treinamento
    def predict_and_train(self, pc: int, outcome: int) -> bool:
        indices = self._get_indices(pc)
        
        # Soma pesos de todas as tabelas
        total_vote = 0
        for i, idx in enumerate(indices):
            total_vote += self.tables[i].get_weight(idx)
        
        # Faz a predição
        prediction = total_vote >= self.threshold
        correct = (prediction == bool(outcome))
        
        # Atualiza apenas em caso de erro de predição
        if not correct:
            error = 1 if outcome else -1
            for i, idx in enumerate(indices):
                self.tables[i].update(idx, error)
        
        # Atualiza históricos
        self._update_histories(pc, outcome)
        
        return correct
    
    # Atualiza todos os registros de histórico
    def _update_histories(self, pc: int, outcome: int):
        # Atualiza histórico global
        self.ghr = np.roll(self.ghr, 1)
        self.ghr[0] = outcome
        
        # Atualiza histórico local
        self.local_history[pc] = np.roll(self.local_history[pc], 1)
        self.local_history[pc][0] = outcome
        
        # Atualiza histórico de caminho
        self.path_history = np.roll(self.path_history, 1)
        self.path_history[0] = pc & 1

# Função principal do programa
def main():
    # Verifica argumentos da linha de comando
    if len(sys.argv) != 12:
        print("Please provide correct number of arguments")
        sys.exit(1)
        
    # Obtém arquivo de entrada e parâmetros
    input_file = sys.argv[1]
    address_size = int(sys.argv[2])
    input_size = sum(int(arg) for arg in sys.argv[3:11]) * 24
    
    # Inicializa o preditor
    predictor = ULEEN(address_size, input_size)
    interval = 10000
    
    try:
        # Processa arquivo de entrada
        with open(input_file, 'r') as f:
            num_branches = 0
            num_predicted = 0
            
            # Loop principal de processamento
            for line in f:
                pc, outcome = map(int, line.strip().split())
                
                # Realiza predição e treinamento
                num_branches += 1
                if predictor.predict_and_train(pc, outcome):
                    num_predicted += 1
                    
                # Imprime resultados parciais
                if num_branches % interval == 0:
                    accuracy = (num_predicted / num_branches) * 100
                    print(f"branch number: {num_branches}")
                    print(f"----- Partial Accuracy: {accuracy:.4f}\n")
            
            # Calcula e imprime resultados finais
            accuracy = (num_predicted / num_branches) * 100
            print("\n----- Results ------")
            print(f"Predicted branches: {num_predicted}")
            print(f"Not predicted branches: {num_branches - num_predicted}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"\n------ Size of ntuple (address_size): {address_size}")
            print(f"------ Size of each input: {input_size}")
            
            # Salva acurácia em arquivo
            with open(f"{input_file}-accuracy.csv", "a") as f:
                f.write(f"{accuracy:.4f} ULEEN\n")
                
    except FileNotFoundError:
        print("Can't open file")
        sys.exit(1)

# Ponto de entrada do programa
if __name__ == "__main__":
    main()