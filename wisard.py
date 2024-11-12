import random
import time
from collections import defaultdict, deque

class ExceptionWiSARD(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg

def calculate_number_of_rams(entry_size, address_size):
    number_of_rams = entry_size // address_size
    remain = entry_size % address_size
    if remain > 0:
        number_of_rams += 1
    return number_of_rams

def save_accuracy(num_branches, num_branches_predicted, filename):
    accuracy = (num_branches_predicted / num_branches) * 100
    path = f"{filename}-accuracy.csv"
    try:
        with open(path, "a") as f:
            f.write(f"{accuracy:.4f} WISARD\n")
    except IOError:
        print("Não foi possível abrir o arquivo para salvar a acurácia.")

class Bleaching:
    @staticmethod
    def make(allvotes):
        labels = [0, 0]
        bleaching = 1
        biggest = 0
        ambiguity = False
        while True:
            for i in range(2):
                labels[i] = 0
                for vote in allvotes[i]:
                    if vote >= bleaching:
                        labels[i] += 1
            bleaching += 1
            biggest = max(labels)
            ambiguity = labels.count(biggest) > 1
            if not ambiguity or biggest <= 1:
                break
        return labels

class RAM:
    def __init__(self, indexes=None):
        self.addresses = indexes or []
        self.positions = defaultdict(int)
        if indexes is not None:
            self.check_limit_address_size(len(indexes))

    def get_vote(self, input_data):
        index = self.get_index(input_data)
        return self.positions.get(index, 0)

    def train(self, input_data):
        index = self.get_index(input_data)
        self.positions[index] += 1

    def get_index(self, input_data):
        index = 0
        p = 1
        for address in self.addresses:
            index += input_data[address] * p
            p *= 2
        return index

    def check_limit_address_size(self, address_size):
        if address_size > 64:
            raise ExceptionWiSARD("The base power to addressSize passed the limit of 2^64!")

class Discriminator:
    def __init__(self, address_size=0, entry_size=0):
        self.entry_size = entry_size
        self.rams = []
        if entry_size > 0:
            self.set_ram_shuffle(address_size)

    def classify(self, input_data):
        votes = [ram.get_vote(input_data) for ram in self.rams]
        return votes

    def train(self, input_data):
        for ram in self.rams:
            ram.train(input_data)

    def set_ram_shuffle(self, address_size):
        self.check_address_size(self.entry_size, address_size)
        number_of_rams = calculate_number_of_rams(self.entry_size, address_size)
        indexes = list(range(self.entry_size))
        random.shuffle(indexes)
        self.rams = [RAM(indexes[i * address_size:(i + 1) * address_size]) for i in range(number_of_rams)]

    def check_address_size(self, entry_size, address_size):
        if address_size < 2 or entry_size < 2 or entry_size < address_size:
            raise ExceptionWiSARD("Invalid address or entry size!")

class Wisard:
    def __init__(self, address_size, input_size):
        self.address_size = address_size
        self.discriminators = [Discriminator(address_size, input_size), Discriminator(address_size, input_size)]

    def train(self, input_data, label):
        self.discriminators[label].train(input_data)

    def classify(self, input_data):
        candidates = self.classify2(input_data)
        return 0 if candidates[0] >= candidates[1] else 1

    def classify2(self, input_data):
        allvotes = [self.discriminators[0].classify(input_data), self.discriminators[1].classify(input_data)]
        return Bleaching.make(allvotes)

# Funções de manipulação de dados e predição de branches
def read_branch(pc, outcome, stream):
    line = stream.readline()
    if not line:
        return False
    data = line.strip().split()
    pc[0] = int(data[0])
    outcome[0] = int(data[1])
    return True

def pc_binary(n):
    return [int(bit) for bit in bin(n)[2:].zfill(32)]

def pc_binary_lower(pc_bits, n):
    return pc_bits[-n:]

def xor_pc_ghr(pc_bits, ghr, n):
    return [pc_bits[i] ^ ghr[i] for i in range(n)]

# Função principal para rodar o programa
def main(argv):
    if len(argv) != 12:
        print("Please, write the file/path_to_file correctly!")
        exit(0)

    stream = open(argv[1], "r")
    if not stream:
        print("Can't open file")
        return

    address_size = int(argv[2])
    params = list(map(int, argv[3:11]))

    pc = [0]
    outcome = [0]
    num_branches = 0
    num_branches_predicted = 0
    ghr = deque([0] * 24, maxlen=24)  # Histórico global com deque
    input_size = sum([params[i] * (24 if i < 3 else (16, 9, 7, 5, 8)[i - 3]) for i in range(8)])

    w = Wisard(address_size, input_size)
    train_data = []

    while read_branch(pc, outcome, stream):
        num_branches += 1

        pc_bits = pc_binary(pc[0])
        pc_bits_lower_24 = pc_binary_lower(pc_bits, 24)
        xor_pc_ghr24 = xor_pc_ghr(pc_bits_lower_24, list(ghr), 24)

        # Montagem dos dados de entrada para treinamento
        train_data = (pc_bits_lower_24 * params[0] +
                      list(ghr) * params[1] +
                      xor_pc_ghr24 * params[2])

        # Adiciona dados adicionais para completar o input_size
        train_data += (pc_bits_lower_24[:16] * params[3] +
                       xor_pc_ghr24[:9] * params[4] +
                       xor_pc_ghr24[:7] * params[5] +
                       xor_pc_ghr24[:5] * params[6] +
                       xor_pc_ghr24[:8] * params[7])

        # Limita o train_data ao tamanho do input_size
        train_data = train_data[:input_size]

        # Realiza a predição
        predicted = w.classify(train_data)
        if predicted == outcome[0]:
            num_branches_predicted += 1

        # Treina o preditor
        w.train(train_data, outcome[0])

        # Atualiza o histórico global
        ghr.append(outcome[0])

        # Exibe precisão parcial a cada 10.000 branches
        if num_branches % 10000 == 0:
            partial_accuracy = (num_branches_predicted / num_branches) * 100
            print(f"Branch number: {num_branches}")
            print(f"----- Partial Accuracy: {partial_accuracy:.2f}%\n")

        train_data.clear()

    # Resultados finais
    final_accuracy = (num_branches_predicted / num_branches) * 100
    print(" ----- Results ------")
    print(f"Predicted branches: {num_branches_predicted}")
    print(f"Not predicted branches: {num_branches - num_branches_predicted}")
    print(f"Accuracy: {final_accuracy:.6f}")
    print(f"\n------ Size of ntuple (address_size): {address_size} -----")
    print(f"\n------ Size of each input: {input_size} -----\n")

    # Salvar a acurácia em um arquivo CSV
    save_accuracy(num_branches, num_branches_predicted, argv[1])

    stream.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 12:
        print("Please, write the file/path_to_file correctly!")
        sys.exit(0)

    main(sys.argv)
