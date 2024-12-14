import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Define label mappings
LABELS = {0: "Contradiction", 1: "Neutral", 2: "Entailment"}

# Function to load data from a JSONL file
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

# Function to count predictions by true label and predicted label
def count_predictions(data):
    # Initialize a nested dictionary to store counts
    counts = {label: {pred: 0 for pred in LABELS.values()} for label in LABELS.values()}

    # Count predictions based on ground truth and predicted labels
    for entry in data:
        true_label = LABELS[entry['ground_truth']]
        predicted_label = LABELS[entry['predicted_label']]
        counts[true_label][predicted_label] += 1

    return counts

# Function to plot the grouped bar chart
def plot_grouped_bar_chart(counts, colors=None):
    labels = list(counts.keys())  # True labels (x-axis categories)
    predicted_labels = list(LABELS.values())  # Predicted label categories

    # Data preparation for grouped bars
    x = range(len(labels))
    width = 0.2  # Width of each bar

    # Default colors if none are provided
    if colors is None:
        colors = ['red', 'blue', 'green']

    # Generate the bars for each predicted label category
    for i, (pred_label, color) in enumerate(zip(predicted_labels, colors)):
        values = [counts[true_label][pred_label] for true_label in labels]
        bar_positions = [pos + i * width for pos in x]
        bars = plt.bar(bar_positions, values, width, label=f"Predicted {pred_label}", color=color)

        # Add y-value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')

    # Configure x-axis and labels
    plt.xlabel("True Labels")
    plt.ylabel("Number of Predictions")
    plt.title("S Predicted Labels on Adversarial Set by True Label")
    plt.xticks([pos + width for pos in x], labels)
    # plt.ylim(0, 3500) 
    plt.legend(loc='best')
    plt.show()

# Main function
def main():
    # Path to the JSONL file
    file_path = "eval_output_snli_on_adv_ex/model_failed_tests.txt"  # Replace with the actual path to your JSONL file

    # Load data
    data = load_data(file_path)

    # Count predictions
    counts = count_predictions(data)

    # Plot grouped bar chart with custom colors (optional)
    custom_colors = ['#478778', '#AFE1AF','#50C878']  # Replace with desired colors or set to None
    plot_grouped_bar_chart(counts, colors=custom_colors)

if __name__ == "__main__":
    main()
