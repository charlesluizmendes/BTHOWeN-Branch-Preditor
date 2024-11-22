import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

def get_latest_files(directory):
    """
    Get the latest CSV file for each type (WISARD, BTHOWeN, ULEEN) from the directory.
    """
    files = {}
    for fname in os.listdir(directory):
        if not fname.endswith('-accuracy.csv'):
            continue
        
        try:
            timestamp = datetime.strptime(fname[:15], '%Y%m%d-%H%M%S')
            if 'WISARD' in fname:
                category = 'WISARD'
            elif 'BTHOWeN' in fname:
                category = 'BTHOWeN'
            elif 'ULEEN' in fname:
                category = 'ULEEN'
            else:
                continue
                
            if category not in files or timestamp > files[category][0]:
                files[category] = (timestamp, os.path.join(directory, fname))
        except ValueError:
            continue
    
    ordered_files = []
    for category in ['WISARD', 'BTHOWeN', 'ULEEN']:
        if category in files:
            ordered_files.append(files[category][1])
    
    if len(ordered_files) != 3:
        raise ValueError(f"Missing required files. Found only {len(ordered_files)} of 3 required types.")
    
    return ordered_files

def plot_csv_data(file1, file2, file3, x_column, y_column, labels, output_file):
    """
    Plot data from three CSV files.
    """
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)
    data3 = pd.read_csv(file3)

    plt.figure(figsize=(10, 6))
    line1, = plt.plot(data1[x_column], data1[y_column], label=labels[0], marker='o')
    line2, = plt.plot(data2[x_column], data2[y_column], label=labels[1], marker='s')
    line3, = plt.plot(data3[x_column], data3[y_column], label=labels[2], marker='^')

    for data, line in [(data1, line1), (data2, line2), (data3, line3)]:
        x = data[x_column].iloc[-1]
        y = data[y_column].iloc[-1]
        plt.text(x, y, f"{y:.2f}%", fontsize=9, ha='left', va='bottom', color=line.get_color())

    plt.title('Comparação de Acurácia vs Número de Ramos Processados')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.grid(True)

    plt.savefig(output_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python graph.py <diretório>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Erro: {directory} não é um diretório válido")
        sys.exit(1)

    try:
        # Get latest files
        files = get_latest_files(directory)
        
        # Create output filename from directory name
        dir_name = os.path.basename(os.path.normpath(directory))
        output_file = f"{dir_name}-accuracy.png"
        
        # Column names and labels
        x_column = "Number of Branches Processed"
        y_column = "Accuracy (%)"
        labels = ["WiSARD", "BTHOWeN", "ULEEN"]

        # Generate plot
        plot_csv_data(*files, x_column, y_column, labels, output_file)
        
    except Exception as e:
        print(f"Erro: {str(e)}")
        sys.exit(1)