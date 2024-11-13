import numpy as np
import sys
from typing import List, Tuple
from collections import defaultdict

class BloomFilter:
    def __init__(self, size: int, num_hash: int):
        """Inicializa o Bloom Filter neural."""
        self.size = size
        self.num_hash = num_hash
        self.weights = np.random.uniform(-1, 1, size)
        self.binary = False
    
    def query(self, pattern: np.ndarray) -> float:
        """Consulta o Bloom Filter e retorna a predição."""
        indices = self._get_indices(pattern)
        if self.binary:
            return 1 if min(self.weights[indices]) > 0 else -1
        return np.tanh(np.mean(self.weights[indices]))
    
    def _get_indices(self, pattern: np.ndarray) -> List[int]:
        """Calcula os índices de hash para o padrão de entrada."""
        indices = []
        for i in range(self.num_hash):
            hash_val = 0
            for j, bit in enumerate(pattern):
                if bit:
                    hash_val ^= (j * 0x9e3779b1 + i)
            indices.append(hash_val % self.size)
        return indices
    
    def update(self, pattern: np.ndarray, gradient: float, learning_rate: float):
        """Atualiza os pesos do Bloom Filter."""
        if not self.binary:
            indices = self._get_indices(pattern)
            self.weights[indices] += learning_rate * gradient
            self.weights = np.clip(self.weights, -1, 1)

class SubModel:
    def __init__(self, input_size: int, table_size: int, num_filters: int = 4):
        """Inicializa um submodelo com múltiplos Bloom Filters."""
        self.input_size = input_size
        self.num_filters = num_filters
        self.filters = [BloomFilter(table_size, num_hash=3) for _ in range(self.num_filters)]
        self.bias = 0.0
    
    def predict(self, pattern: np.ndarray) -> float:
        """Realiza a predição combinando os resultados dos Bloom Filters."""
        total = sum(f.query(pattern) for f in self.filters) + self.bias
        return total

class ULEEN:   
    def __init__(self, address_size: int, feature_config: List[int]):
        """
        Inicializa o preditor ULEEN.
        
        Args:
            address_size: Tamanho do endereço em bits
            feature_config: Lista de configurações [ghr_bits, local_bits, path_bits, 
                                                 num_submodels, num_hash, dropout_rate]
        """
        # Configuração de features
        self.pc_bits = 16
        self.ghr_bits = feature_config[0] if len(feature_config) > 0 else 16
        self.local_bits = feature_config[1] if len(feature_config) > 1 else 8
        self.path_bits = feature_config[2] if len(feature_config) > 2 else 8
        
        # Calcula tamanho total do input
        self.input_size = self.pc_bits + self.ghr_bits + self.local_bits + self.path_bits
        
        # Configuração do modelo
        self.table_size = 2**14
        self.num_submodels = feature_config[3] if len(feature_config) > 3 else 3
        self.num_hash = feature_config[4] if len(feature_config) > 4 else 3
        self.dropout_rate = float(feature_config[5])/100 if len(feature_config) > 5 else 0.5
        
        # Parâmetros de aprendizado
        self.learning_rate = 0.01
        self.threshold = 0.0
        
        # Inicializa submodelos
        self.submodels = [
            SubModel(self.input_size, self.table_size) 
            for _ in range(self.num_submodels)
        ]
        
        # Inicializa registradores de história
        self.ghr = np.zeros(max(32, self.ghr_bits), dtype=np.int32)
        self.local_history = defaultdict(lambda: np.zeros(max(16, self.local_bits), dtype=np.int32))
        self.path_history = np.zeros(max(16, self.path_bits), dtype=np.int32)
        
        # Perceptron backup
        self.perceptron_weights = np.zeros(self.input_size)
        self.perceptron_threshold = 2.14 * np.sqrt(self.input_size)
        
        # Ajuste dinâmico de threshold
        self.threshold_counter = 0
        self.threshold_window = 1000
        self.min_threshold = -1.0
        self.max_threshold = 1.0
        
        # Métricas de performance
        self.recent_predictions = []
        self.max_recent = 10000
    
    def _preprocess_input(self, pc: int) -> np.ndarray:
        """Prepara o vetor de features para predição."""
        features = []
        features.extend(self._to_binary(pc, self.pc_bits))
        features.extend(self.ghr[:self.ghr_bits])
        features.extend(self.local_history[pc][:self.local_bits])
        features.extend(self.path_history[:self.path_bits])
        return np.array(features, dtype=np.int32)
    
    def _to_binary(self, value: int, bits: int) -> List[int]:
        """Converte um valor para sua representação binária."""
        return [(value >> i) & 1 for i in range(bits)]
    
    def _get_confidence(self, predictions: List[float]) -> float:
        """Calcula a confiança da predição."""
        if not predictions:
            return 0.0
        mean_pred = np.mean(predictions)
        return abs(mean_pred)
    
    def predict_and_train(self, pc: int, outcome: int) -> bool:
        """
        Realiza predição e treinamento para um branch.
        
        Args:
            pc: Program Counter
            outcome: Resultado real do branch (0 ou 1)
            
        Returns:
            bool: True se a predição estava correta
        """
        pattern = self._preprocess_input(pc)
        
        # Predição principal usando Bloom Filters
        active_models = np.random.random(self.num_submodels) >= self.dropout_rate
        predictions = []
        for i, model in enumerate(self.submodels):
            if active_models[i]:
                predictions.append(model.predict(pattern))
        
        # Predição do perceptron backup
        perceptron_output = np.dot(self.perceptron_weights, pattern)
        use_perceptron = not predictions
        
        # Decisão final
        if use_perceptron:
            prediction = perceptron_output >= self.perceptron_threshold
            confidence = abs(perceptron_output) / self.perceptron_threshold
        else:
            total_vote = np.mean(predictions)
            prediction = total_vote >= self.threshold
            confidence = self._get_confidence(predictions)
        
        correct = (prediction == bool(outcome))
        
        # Atualização dos modelos
        if not correct or confidence < 0.5:  # Margin training
            target = 1 if outcome else -1
            
            # Treina modelos principais
            if not use_perceptron:
                gradient = target - np.mean(predictions)
                for i, model in enumerate(self.submodels):
                    if active_models[i]:
                        for filter in model.filters:
                            filter.update(pattern, gradient, self.learning_rate)
                        model.bias += self.learning_rate * gradient
            
            # Treina perceptron backup
            if use_perceptron or not correct:
                perceptron_gradient = target - np.sign(perceptron_output)
                self.perceptron_weights += self.learning_rate * perceptron_gradient * pattern
        
        # Ajuste dinâmico de threshold
        self.threshold_counter += 1
        if self.threshold_counter >= self.threshold_window:
            self._adjust_threshold()
            self.threshold_counter = 0
        
        # Atualiza históricos
        self._update_histories(pc, outcome)
        
        # Mantém registro de predições recentes
        self.recent_predictions.append(correct)
        if len(self.recent_predictions) > self.max_recent:
            self.recent_predictions.pop(0)
        
        return correct
    
    def _adjust_threshold(self):
        """Ajusta o threshold baseado na performance recente."""
        recent_accuracy = np.mean(self.recent_predictions)
        if recent_accuracy < 0.98:
            self.threshold *= 0.95
        else:
            self.threshold *= 1.05
        self.threshold = np.clip(self.threshold, self.min_threshold, self.max_threshold)
    
    def _update_histories(self, pc: int, outcome: int):
        """Atualiza os registradores de história."""
        # Atualiza história global
        self.ghr = np.roll(self.ghr, 1)
        self.ghr[0] = outcome
        
        # Atualiza história local
        self.local_history[pc] = np.roll(self.local_history[pc], 1)
        self.local_history[pc][0] = outcome
        
        # Atualiza história de caminho
        self.path_history = np.roll(self.path_history, 1)
        self.path_history[0] = pc & 1

def main():
    if len(sys.argv) < 3:
        print("Usage: script.py <trace_file> <address_size> [ghr_bits] [local_bits] [path_bits] [num_submodels] [num_hash] [dropout_rate]")
        sys.exit(1)
    
    try:
        input_file = sys.argv[1]
        address_size = int(sys.argv[2])
        
        # Parse configurações opcionais
        feature_config = []
        for i in range(3, len(sys.argv)):
            feature_config.append(int(sys.argv[i]))
        
        predictor = ULEEN(address_size, feature_config)
        interval = 10000
        
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
            
            # Calcula e imprime resultados finais
            accuracy = (num_predicted / num_branches) * 100
            print("\n----- Results ------")
            print(f"Predicted branches: {num_predicted}")
            print(f"Not predicted branches: {num_branches - num_predicted}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"\n------ Size of ntuple (address_size): {address_size}")
            print(f"------ Size of each input: {predictor.input_size}")
            
            # Salva acurácia em arquivo
            with open(f"{input_file}-accuracy.csv", "a") as f:
                f.write(f"{accuracy:.4f} ULEEN\n")
    
    except FileNotFoundError:
        print("Can't open file")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()