import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


# Function for bar plots 
def BarPlot(column):
    # get value counts for column
    value_counts = column.value_counts()

    # specify Tableau colors for bars
    colors = ['#0072B2', '#FDBF6F', '#009E73']

    # create bar chart using Matplotlib with specified colors
    plt.bar(value_counts.index, value_counts.values, color=colors)

    # set title and axis labels
    plt.title(f'{column.name}')
    plt.xlabel(f'{column.name} Values')
    plt.ylabel('Frequency')

    # create legend outside of chart
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
    labels = value_counts.index

    # display chart
    plt.show()


# Evaluation function that prints classification report and confusion matrix
def evaluate(y_test, y_pred, model_name):
    #print accuracy
    cm = confusion_matrix(y_test, y_pred)

    #ratio of true positive to false positive
    cm_ratio = cm[1,1]/cm[0,1]

    print(f"\033[34m{model_name}\033[0m")
    # print(cm[1,1]) in f string true positive
    print(f'Ratio of true positive to false positive: {cm_ratio:.2f}')
    print()
    print(f" \033[32mClassification Report:\033[0m")
    print(classification_report(y_test, y_pred))
    print(f" \033[32mConfusion Matrix:\033[0m")
    
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title(f'{model_name}: Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

def plot_roc_with_cost_ratio(y_test, y_pred_rf, cost_ratio):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf)
    plt.plot(fpr, tpr, linewidth=2, label="Random Forest")
    
    plt.plot([0, 1], [0, cost_ratio], 'r-', label="Cost Ratio")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()


# Define a function to apply the color
def color_above_threshold(val, threshold):
    color = 'green' if val > threshold else 'red'
    return f'color: {color}'

def print_votes_and_prediction(model_votes, threshold):
    # Create an empty list to store the rows of the table
    table_rows = []

    # Loop over the model votes
    for model, predicted_probability in model_votes:
        # Create a dictionary for this row
        row = {
            "Model": model.__class__.__name__,
            "Probability": predicted_probability
        }
        # Add the dictionary to the list
        table_rows.append(row)

    # Create a DataFrame from the list of dictionaries
    votes_table = pd.DataFrame(table_rows)

    #print fstring for threshold
    print(f"Threshold: {threshold}")
    print("Model Votes:")

    # Apply the color to the 'Probability' column
    styled_table = votes_table.style.applymap(lambda val: color_above_threshold(val, threshold), subset=['Probability'])
    display(styled_table)

