import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_csv_data(file1, file2, file3, x_column, y_column, labels, output_file=None):
    """
    Plots data from three CSV files.

    :param file1: Path to the first CSV file
    :param file2: Path to the second CSV file
    :param file3: Path to the third CSV file
    :param x_column: Name of the column to use for the X-axis
    :param y_column: Name of the column to use for the Y-axis
    :param labels: List of labels for the three datasets
    :param output_file: Path to save the plot as an image (optional)
    """
    # Load the data
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)
    data3 = pd.read_csv(file3)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(data1[x_column], data1[y_column], label=labels[0], marker='o')
    plt.plot(data2[x_column], data2[y_column], label=labels[1], marker='s')
    plt.plot(data3[x_column], data3[y_column], label=labels[2], marker='^')

    # Customize the plot
    plt.title('Comparison of Accuracy vs Number of Branches Processed')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.grid(True)

    # Show or save the plot
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python graph.py <wisard.csv> <bthowen.csv> <uleen.csv> [output_file]")
        sys.exit(1)

    # Parse command-line arguments
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]
    output_file = sys.argv[4] if len(sys.argv) > 4 else None

    # Column names and labels
    x_column = "Number of Branches Processed"
    y_column = "Accuracy (%)"
    labels = ["WiSARD", "BTHOWeN", "ULEEN"]

    # Generate the plot
    plot_csv_data(file1, file2, file3, x_column, y_column, labels, output_file)
