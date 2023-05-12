import pandas as pd
from IPython.display import display

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
