import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_csv_data(file1, file2, file3, x_column, y_column, labels, output_file=None):
    """
    Plota dados de três arquivos CSV.

    :param file1: Caminho para o primeiro arquivo CSV
    :param file2: Caminho para o segundo arquivo CSV
    :param file3: Caminho para o terceiro arquivo CSV
    :param x_column: Nome da coluna para o eixo X
    :param y_column: Nome da coluna para o eixo Y
    :param labels: Lista de rótulos para os três conjuntos de dados
    :param output_file: Caminho para salvar o gráfico como imagem (opcional)
    """
    # Carregar os dados
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)
    data3 = pd.read_csv(file3)

    # Plotar os dados
    plt.figure(figsize=(10, 6))
    line1, = plt.plot(data1[x_column], data1[y_column], label=labels[0], marker='o')
    line2, = plt.plot(data2[x_column], data2[y_column], label=labels[1], marker='s')
    line3, = plt.plot(data3[x_column], data3[y_column], label=labels[2], marker='^')

    # Adicionar anotações com porcentagens e casas decimais no final de cada linha
    # Conjunto de dados 1
    x1 = data1[x_column].iloc[-1]
    y1 = data1[y_column].iloc[-1]
    plt.text(x1, y1, f"{y1:.2f}%", fontsize=9, ha='left', va='bottom', color=line1.get_color())

    # Conjunto de dados 2
    x2 = data2[x_column].iloc[-1]
    y2 = data2[y_column].iloc[-1]
    plt.text(x2, y2, f"{y2:.2f}%", fontsize=9, ha='left', va='bottom', color=line2.get_color())

    # Conjunto de dados 3
    x3 = data3[x_column].iloc[-1]
    y3 = data3[y_column].iloc[-1]
    plt.text(x3, y3, f"{y3:.2f}%", fontsize=9, ha='left', va='bottom', color=line3.get_color())

    # Personalizar o gráfico
    plt.title('Comparação de Acurácia vs Número de Ramos Processados')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.grid(True)

    # Mostrar ou salvar o gráfico
    if output_file:
        plt.savefig(output_file)
        print(f"Gráfico salvo em {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: python graph.py <wisard.csv> <bthowen.csv> <uleen.csv> [output_file]")
        sys.exit(1)

    # Analisar argumentos de linha de comando
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]
    output_file = sys.argv[4] if len(sys.argv) > 4 else None

    # Nomes das colunas e rótulos
    x_column = "Number of Branches Processed"
    y_column = "Accuracy (%)"
    labels = ["WiSARD", "BTHOWeN", "ULEEN"]

    # Gerar o gráfico
    plot_csv_data(file1, file2, file3, x_column, y_column, labels, output_file)
