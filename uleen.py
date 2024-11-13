import numpy as np 
import sys  
from typing import List, Tuple 
from collections import defaultdict  

class BloomFilter:
    def __init__(self, size: int, num_hash: int):
        """Inicializa o Bloom Filter neural."""
        self.size = size  # Tamanho do array do Bloom filter
        self.num_hash = num_hash  # Número de funções hash a serem utilizadas
        self.weights = np.random.uniform(-1, 1, size)  # Inicializa pesos aleatórios entre -1 e 1
        self.binary = False  # Flag para determinar se o filtro deve retornar valores binários
    
    def query(self, pattern: np.ndarray) -> float:
        """Consulta o Bloom Filter e retorna a predição."""
        indices = self._get_indices(pattern)  # Obtém índices hash para o padrão de entrada
        if self.binary:  # Se o modo binário está ativado
            return 1 if min(self.weights[indices]) > 0 else -1  # Retorna predição binária
        return np.tanh(np.mean(self.weights[indices]))  # Retorna predição contínua usando tanh
    
    def _get_indices(self, pattern: np.ndarray) -> List[int]:
        """Calcula os índices de hash para o padrão de entrada."""
        indices = []  # Lista para armazenar índices hash
        for i in range(self.num_hash):  # Para cada função hash
            hash_val = 0  # Inicializa valor hash
            for j, bit in enumerate(pattern):  # Para cada bit no padrão
                if bit:  # Se o bit é 1
                    hash_val ^= (j * 0x9e3779b1 + i)  # XOR com número mágico para melhor distribuição
            indices.append(hash_val % self.size)  # Adiciona índice dentro do tamanho do filtro
        return indices
    
    def update(self, pattern: np.ndarray, gradient: float, learning_rate: float):
        """Atualiza os pesos do Bloom Filter."""
        if not self.binary:  # Atualiza apenas se não estiver em modo binário
            indices = self._get_indices(pattern)  # Obtém índices para o padrão
            self.weights[indices] += learning_rate * gradient  # Atualiza pesos usando descida do gradiente
            self.weights = np.clip(self.weights, -1, 1)  # Limita pesos entre [-1, 1]

class SubModel:
    def __init__(self, input_size: int, table_size: int, num_filters: int = 4):
        """Inicializa um submodelo com múltiplos Bloom Filters."""
        self.input_size = input_size  # Tamanho dos padrões de entrada
        self.num_filters = num_filters  # Número de Bloom filters no submodelo
        self.filters = [BloomFilter(table_size, num_hash=3) for _ in range(self.num_filters)]  # Cria Bloom filters
        self.bias = 0.0  # Inicializa termo de viés
    
    def predict(self, pattern: np.ndarray) -> float:
        """Realiza a predição combinando os resultados dos Bloom Filters."""
        total = sum(f.query(pattern) for f in self.filters) + self.bias  # Soma predições de todos os filtros mais viés
        return total

class ULEEN:   
    def __init__(self, address_size: int, feature_config: List[int]):
        """Inicializa o preditor ULEEN."""
        # Configuração de características
        self.pc_bits = 16  # Número de bits do contador de programa
        self.ghr_bits = feature_config[0] if len(feature_config) > 0 else 16  # Bits do registrador de histórico global
        self.local_bits = feature_config[1] if len(feature_config) > 1 else 8  # Bits de histórico local
        self.path_bits = feature_config[2] if len(feature_config) > 2 else 8  # Bits de histórico de caminho
        
        # Calcula tamanho total da entrada
        self.input_size = self.pc_bits + self.ghr_bits + self.local_bits + self.path_bits  # Tamanho total do vetor de características
        
        # Configuração do modelo
        self.table_size = 2**14  # Tamanho das tabelas hash
        self.num_submodels = feature_config[3] if len(feature_config) > 3 else 3  # Número de submodelos
        self.num_hash = feature_config[4] if len(feature_config) > 4 else 3  # Número de funções hash
        self.dropout_rate = float(feature_config[5])/100 if len(feature_config) > 5 else 0.5  # Probabilidade de dropout
        
        # Parâmetros de aprendizado
        self.learning_rate = 0.01  # Taxa de aprendizado para descida do gradiente
        self.threshold = 0.0  # Limiar de decisão
        
        # Inicializa submodelos
        self.submodels = [
            SubModel(self.input_size, self.table_size) 
            for _ in range(self.num_submodels)
        ]  # Cria array de submodelos
        
        # Inicializa registradores de histórico
        self.ghr = np.zeros(max(32, self.ghr_bits), dtype=np.int32)  # Registrador de histórico global
        self.local_history = defaultdict(lambda: np.zeros(max(16, self.local_bits), dtype=np.int32))  # Tabela de histórico local
        self.path_history = np.zeros(max(16, self.path_bits), dtype=np.int32)  # Registrador de histórico de caminho
        
        # Perceptron de backup
        self.perceptron_weights = np.zeros(self.input_size)  # Pesos para perceptron de backup
        self.perceptron_threshold = 2.14 * np.sqrt(self.input_size)  # Limiar do perceptron
        
        # Ajuste dinâmico de limiar
        self.threshold_counter = 0  # Contador para atualizações de limiar
        self.threshold_window = 1000  # Tamanho da janela para ajuste de limiar
        self.min_threshold = -1.0  # Valor mínimo do limiar
        self.max_threshold = 1.0  # Valor máximo do limiar
        
        # Métricas de desempenho
        self.recent_predictions = []  # Lista de resultados de predições recentes
        self.max_recent = 10000  # Número máximo de predições recentes para armazenar
    
    def _preprocess_input(self, pc: int) -> np.ndarray:
        """Prepara o vetor de características para predição."""
        features = []  # Inicializa vetor de características
        features.extend(self._to_binary(pc, self.pc_bits))  # Adiciona bits do PC
        features.extend(self.ghr[:self.ghr_bits])  # Adiciona histórico global
        features.extend(self.local_history[pc][:self.local_bits])  # Adiciona histórico local
        features.extend(self.path_history[:self.path_bits])  # Adiciona histórico de caminho
        return np.array(features, dtype=np.int32)
    
    def _to_binary(self, value: int, bits: int) -> List[int]:
        """Converte um valor para sua representação binária."""
        return [(value >> i) & 1 for i in range(bits)]  # Converte inteiro para array binário
    
    def _get_confidence(self, predictions: List[float]) -> float:
        """Calcula a confiança da predição."""
        if not predictions:  # Se não há predições disponíveis
            return 0.0
        mean_pred = np.mean(predictions)  # Calcula média das predições
        return abs(mean_pred)  # Retorna valor absoluto como confiança
    
    def predict_and_train(self, pc: int, outcome: int) -> bool:
        """Realiza predição e treinamento para um branch."""
        pattern = self._preprocess_input(pc)  # Prepara características de entrada
        
        # Predição principal usando Bloom Filters
        active_models = np.random.random(self.num_submodels) >= self.dropout_rate  # Aplica dropout
        predictions = []  # Armazena predições dos modelos ativos
        for i, model in enumerate(self.submodels):  # Para cada submodelo
            if active_models[i]:  # Se o modelo está ativo
                predictions.append(model.predict(pattern))  # Adiciona sua predição
        
        # Predição do perceptron de backup
        perceptron_output = np.dot(self.perceptron_weights, pattern)  # Calcula saída do perceptron
        use_perceptron = not predictions  # Usa perceptron se não há modelos ativos
        
        # Toma decisão final
        if use_perceptron:  # Se usando perceptron
            prediction = perceptron_output >= self.perceptron_threshold  # Faz predição do perceptron
            confidence = abs(perceptron_output) / self.perceptron_threshold  # Calcula confiança
        else:  # Se usando Bloom filters
            total_vote = np.mean(predictions)  # Média das predições
            prediction = total_vote >= self.threshold  # Faz predição do conjunto
            confidence = self._get_confidence(predictions)  # Calcula confiança
        
        correct = (prediction == bool(outcome))  # Verifica se predição estava correta
        
        # Atualiza modelos
        if not correct or confidence < 0.5:  # Se incorreto ou baixa confiança
            target = 1 if outcome else -1  # Converte resultado para valor alvo
            
            # Treina modelos principais
            if not use_perceptron:  # Se usando Bloom filters
                gradient = target - np.mean(predictions)  # Calcula gradiente
                for i, model in enumerate(self.submodels):  # Atualiza cada modelo ativo
                    if active_models[i]:
                        for filter in model.filters:  # Atualiza cada filtro
                            filter.update(pattern, gradient, self.learning_rate)
                        model.bias += self.learning_rate * gradient  # Atualiza viés
            
            # Treina perceptron de backup
            if use_perceptron or not correct:  # Se usando perceptron ou predição errada
                perceptron_gradient = target - np.sign(perceptron_output)  # Calcula gradiente do perceptron
                self.perceptron_weights += self.learning_rate * perceptron_gradient * pattern  # Atualiza pesos
        
        # Atualiza limiar
        self.threshold_counter += 1  # Incrementa contador
        if self.threshold_counter >= self.threshold_window:  # Se janela completa
            self._adjust_threshold()  # Ajusta limiar
            self.threshold_counter = 0  # Reinicia contador
        
        # Atualiza históricos
        self._update_histories(pc, outcome)  # Atualiza todos os registradores de histórico
        
        # Acompanha desempenho recente
        self.recent_predictions.append(correct)  # Adiciona resultado ao histórico
        if len(self.recent_predictions) > self.max_recent:  # Se histórico muito longo
            self.recent_predictions.pop(0)  # Remove entrada mais antiga
        
        return correct
    
    def _adjust_threshold(self):
        """Ajusta o limiar baseado no desempenho recente."""
        recent_accuracy = np.mean(self.recent_predictions)  # Calcula acurácia recente
        if recent_accuracy < 0.98:  # Se acurácia abaixo do alvo
            self.threshold *= 0.95  # Diminui limiar
        else:  # Se acurácia acima do alvo
            self.threshold *= 1.05  # Aumenta limiar
        self.threshold = np.clip(self.threshold, self.min_threshold, self.max_threshold)  # Mantém limiar dentro dos limites
    
    def _update_histories(self, pc: int, outcome: int):
        """Atualiza os registradores de histórico."""
        # Atualiza histórico global
        self.ghr = np.roll(self.ghr, 1)  # Desloca registrador de histórico
        self.ghr[0] = outcome  # Adiciona novo resultado
        
        # Atualiza histórico local
        self.local_history[pc] = np.roll(self.local_history[pc], 1)  # Desloca histórico local
        self.local_history[pc][0] = outcome  # Adiciona novo resultado
        
        # Atualiza histórico de caminho
        self.path_history = np.roll(self.path_history, 1)  # Desloca histórico de caminho
        self.path_history[0] = pc & 1  # Adiciona novo bit de caminho

def main():
    if len(sys.argv) < 3:  # Verifica argumentos mínimos necessários
        print("Uso: script.py <arquivo_trace> <tamanho_endereco> [ghr_bits] [local_bits] [path_bits] [num_submodels] [num_hash] [dropout_rate]")
        sys.exit(1)
    
    try:
        input_file = sys.argv[1]  # Obtém caminho do arquivo de entrada
        address_size = int(sys.argv[2])  # Obtém tamanho do endereço
        
        # Analisa configurações opcionais
        feature_config = []  # Lista para parâmetros opcionais
        for i in range(3, len(sys.argv)):  # Processa argumentos adicionais
            feature_config.append(int(sys.argv[i]))
        
        predictor = ULEEN(address_size, feature_config)  # Cria preditor
        interval = 10000  # Intervalo de relatório de progresso
        
        with open(input_file, 'r') as f:  # Abre arquivo de trace
            num_branches = 0  # Total de branches processados
            num_predicted = 0  # Branches corretamente preditos
            
            for line in f:  # Processa cada linha
                pc, outcome = map(int, line.strip().split())  # Analisa PC e resultado
                
                num_branches += 1  # Incrementa contagem total
                if predictor.predict_and_train(pc, outcome):  # Faz predição e treina
                    num_predicted += 1  # Incrementa predições corretas
                
                if num_branches % interval == 0:  # Se no intervalo de relatório
                    accuracy = (num_predicted / num_branches) * 100  # Calcula precisão parcial
                    print(f"branch number: {num_branches}")  # Mostra número atual de branches
                    print(f"----- Partial Accuracy: {accuracy:.4f}\n")  # Mostra precisão parcial formatada
            
            # Calcula e imprime resultados finais
            accuracy = (num_predicted / num_branches) * 100  # Calcula precisão final
            print("\n----- Results ------")  # Imprime cabeçalho dos resultados
            print(f"Predicted branches: {num_predicted}")  # Mostra número total de predições corretas
            print(f"Not predicted branches: {num_branches - num_predicted}")  # Mostra número de predições incorretas
            print(f"Accuracy: {accuracy:.4f}")  # Mostra precisão final formatada
            print(f"\n------ Size of ntuple (address_size): {address_size}")  # Mostra tamanho do endereço usado
            print(f"------ Size of each input: {predictor.input_size}")  # Mostra tamanho total da entrada
            
            # Salva resultados em arquivo
            with open(f"{input_file}-accuracy.csv", "a") as f:  # Abre arquivo de resultados em modo append
                f.write(f"{accuracy:.4f} ULEEN\n")  # Escreve precisão final com identificador do modelo
    
    except FileNotFoundError:  # Trata erro de arquivo não encontrado
        print("Can't open file")  # Mostra mensagem de erro
        sys.exit(1)  # Encerra programa com código de erro
    except Exception as e:  # Trata outros erros possíveis
        print(f"Error: {str(e)}")  # Mostra mensagem de erro detalhada
        sys.exit(1)  # Encerra programa com código de erro

if __name__ == "__main__":  # Verifica se é o arquivo principal
    main()  # Executa função principal