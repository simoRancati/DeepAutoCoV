import pandas as pd
import os
import tensorflow as tf
import numpy as np
from collections import Counter
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import csv
from Bio import Align
from Bio.Align import substitution_matrices


def plot_evaluation_metrics(mean_positions, mrrs, precisions_top_25_percent, precisions_top_50_percent,
                            precisions_top_75_percent, save_path):
    """
    Plots the evaluation metrics over iterations and saves the plot to a specified path.

    Parameters:
    mean_positions (list or array): Mean positions over iterations.
    mrrs (list or array): Mean Reciprocal Ranks (MRR) over iterations.
    precisions_top_25_percent (list or array): Precision in the top 25% over iterations.
    precisions_top_50_percent (list or array): Precision in the top 50% over iterations.
    precisions_top_75_percent (list or array): Precision in the top 75% over iterations.
    save_path (str): Path to save the plot.
    """

    plt.figure(figsize=(10, 6))
    plt.plot(mrrs, label="MRR")
    plt.plot(precisions_top_25_percent, label="Precision Top 25%")
    plt.plot(precisions_top_50_percent, label="Precision Top 50%")
    plt.plot(precisions_top_75_percent, label="Precision Top 75%")
    plt.xlabel("Iteration")
    plt.ylabel("Metrics")
    plt.title("Evaluation Metrics Over Iterations")
    plt.legend()
    plt.savefig(save_path)


def analyze_lineages(lineages, mse_list, dominant_lineages):
    """
    This function analyzes the provided lineages and their respective MSE values,
    sorts them in descending order of MSE, calculates the positions of dominant
    lineages and their sublineages, and computes various evaluation metrics.

    Args:
    - lineages (list): List of lineage identifiers.
    - mse_list (list): List of Mean Squared Error (MSE) values corresponding to the lineages.
    - dominant_lineages (list): List of dominant lineage identifiers.

    Returns:
    - dict: A dictionary containing sorted lineages, sorted MSE values,
            mean position of dominant lineages, Mean Reciprocal Rank (MRR),
            and precision values for the top 25%, 50%, and 75%.
    """

    # Combine lineages and mse_list into a list of tuples
    combined = list(zip(lineages, mse_list))

    # Sort the combined list based on MSE in descending order
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)

    # Unzip the sorted combined list back into separate lists
    sorted_lineages, sorted_mse_list = zip(*sorted_combined) if combined else ([], [])

    # Find the positions of dominant lineages and their sublineages
    positions = []
    for dominant in dominant_lineages:
        for i, lineage in enumerate(sorted_lineages):
            if lineage == dominant or lineage.startswith(dominant + '.'):
                positions.append(i + 1)  # Use 1-based indexing for positions

    # Calculate the mean position of the dominant lineages
    mean_position = sum(positions) / len(positions) if positions else None # 1-based indexing for positions

    # Calculate the Mean Reciprocal Rank (MRR)
    reciprocal_ranks = [1 / pos for pos in positions]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None

    # Function to calculate precision for a given threshold
    def calculate_precision(positions, threshold):
        count_in_top = sum(1 for pos in positions if pos <= threshold + 1)
        return count_in_top / (threshold + 1)

    # Calculate precision for top 25%, 50%, and 75%
    total_count = len(sorted_lineages)
    precision_top_25 = calculate_precision(positions, int(total_count * 0.25))
    precision_top_50 = calculate_precision(positions, int(total_count * 0.50))
    precision_top_75 = calculate_precision(positions, int(total_count * 0.75))

    return {
        "sorted_lineages": list(sorted_lineages),
        "sorted_mse_list": list(sorted_mse_list),
        "mean_position": mean_position,
        "mrr": mrr,
        "precision_top_25": precision_top_25,
        "precision_top_50": precision_top_50,
        "precision_top_75": precision_top_75,
    }

def update_class_list(df, id_list, class_list):
    """
    This function updates the class list based on information provided in a DataFrame.

    Parameters:
    - df: DataFrame containing 'id' and 'sequence_last_class' columns.
    - id_list: List of IDs for which the class needs to be updated.
    - class_list: Original list of classes to be updated.

    Returns:
    - An updated list of classes where, for each ID in id_list, the corresponding class in class_list
      has been updated with the value from the 'sequence_last_class' column of the DataFrame,
      if the ID is present in the DataFrame.
    """

    # Create a dictionary from the CSV to map IDs to 'last class'
    id_to_last_class = dict(zip(df['id'], df['sequence_last_class']))

    # Update the class_list based on the 'last class' in the CSV for the given IDs
    for i, id_ in enumerate(id_list):
        if id_ in id_to_last_class:
            class_list[i] = id_to_last_class[id_]

    return class_list

def adjust_predictions_based_on_blosum(blosum_scores, predictions, threshold):
    """
    Adjust predictions based on Blosum scores and a given threshold.
    If a Blosum score is less than the threshold, change the prediction to 1 (true positive).

    :param blosum_scores: List of Blosum scores corresponding to predictions.
    :param predictions: List of predictions (typically -1 for false positive, 1 for true positive).
    :param threshold: Blosum score threshold below which a prediction is set to 1.
    :return: Modified list of predictions.
    """
    adjusted_predictions = []

    # Iterate through both scores and predictions simultaneously
    for score, prediction in zip(blosum_scores, predictions):
        if score > threshold and prediction == -1:
            adjusted_predictions.append(1)
        else:
            adjusted_predictions.append(prediction)

    return adjusted_predictions


def count_true_and_false_positives_with_blosum_score(df, known_lineages):
    """
    Counts the true positives and false positives based on a DataFrame
    with 'Lineages_TP' and 'Blosum_score' columns. It calculates the average Blosum score
    for true positives.

    A true positive is a lineage that is either present in the list of known lineages
    or is a sublineage of a known element. A false positive is a lineage that does not
    meet the criteria to be a true positive.

    :param df: DataFrame containing 'Lineages_TP' and 'Blosum_score'.
    :param known_lineages: List of known lineages.
    :return: Average Blosum score for true positives.
    """
    total_blosum_score_tp = 0

    # Clean up spaces in the known lineages list
    cleaned_known_lineages = [lineage.strip() for lineage in known_lineages]
    total_blosum_score_tp = []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        predicted = row['Lineages_TP'].strip()
        if any(predicted.startswith(known) for known in cleaned_known_lineages):
            total_blosum_score_tp.append(row['Blosum_score'])

    # Calculate the average Blosum score for true positives if there are any true positives
    median_blosum_tp = np.median(total_blosum_score_tp)

    return  median_blosum_tp

def origin_spike(dict_seq):
    """
        Return Spike Protein isolated in Whuan.
    """
    seq_orig = dict_seq['EPI_ISL_402123']

    return seq_orig


def calculate_blosum_scores(reference, sequences):
    """
    Calculate BLOSUM62 alignment scores between a reference sequence and a list of sequences.

    Input:
    - reference: A string representing the reference protein sequence.
    - sequences: A list of strings representing the protein sequences to be compared against the reference.

    Output:
    - scores: A list of floats representing the alignment scores for each sequence in the 'sequences' list,
              maintaining the order of the input sequences.
    """
    # Load matrix
    matrix = substitution_matrices.load("DAYHOFF")

    # Create a PairwiseAligner object
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = matrix

    # List to store the scores
    scores = []

    # Perform global alignment for each sequence and store the score
    for sequence in sequences:
        score = aligner.score(reference.rstrip("*"), sequence.rstrip("*"))
        scores.append(score)

    # Return the list of scores
    return np.array(scores)

def read_protein_sequences_header(file):
    """
    This function reads a FASTA file, extracts protein sequences, and their respective headers,
    and stores them in a dictionary with headers as keys and sequences as values.

    Parameters:
    - file: The path to the FASTA file.

    Returns:
    - sequences_dict: A dictionary where each key is a header and the corresponding value is the protein sequence.
    """

    sequences = []  # Initialize an empty list to store protein sequences.
    headers = []    # Initialize an empty list to store headers of the sequences.
    sequences_dict = {}  # Initialize an empty dictionary to store the sequences with headers as keys.

    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        current_sequence = ''  # A variable to hold the current sequence.

        for line in f:
            line = line.strip()  # Strip whitespace from the line.

            if line.startswith('>'):  # Check for header line.
                if current_sequence:  # If there's a current sequence, store it before starting a new one.
                    sequences.append(current_sequence)
                    headers.append(current_header)  # Store the current header
                    sequences_dict[current_header] = current_sequence  # Map the current header to its sequence
                    current_sequence = ''

                # Update the current header, excluding the first character ('>')
                current_header = line[1:]

            else:
                current_sequence += line  # Append non-header lines to the current sequence.

        if current_sequence:  # After the last line, add the final sequence.
            sequences.append(current_sequence)
            headers.append(current_header)
            sequences_dict[current_header] = current_sequence  # Map the last header to its sequence

    return sequences_dict

def load_data(dir_dataset, week_range):
    """
    This function loads data from a specified directory for a given range of weeks.
    Parameters:
    - dir_dataset: The directory containing the dataset.
    - week_range: A list of weeks for which the data is to be loaded.

    Returns:
    - A concatenated DataFrame containing data from all specified weeks.
    - A list containing the week labels for each row in the DataFrame.
    """

    # Convert the week numbers to strings for matching with folder names.
    week_range = [str(x) for x in week_range]

    # List all folders in dir_dataset that match the specified week range.
    weeks_folder = [x for x in os.listdir(dir_dataset) if x in week_range]

    # Initialize empty lists to store DataFrames and week labels.
    df_list = []
    w_list = []

    # Iterate through each folder corresponding to a week.
    for week in weeks_folder:
        # Construct the path to the dataset file for that week.
        df_path = dir_dataset + week + '/week_dataset.txt'

        # Load the dataset into a pandas DataFrame.
        df = pd.read_csv(df_path, header=None)

        # Append the DataFrame to df_list and replicate the week label for each row in the DataFrame.
        df_list.append(df)
        w_list += [week] * df.shape[0]

        # The following block seems to be a template for further file processing.
        # It sets a directory path and iterates through its files, but no specific operations are performed.
        directory = os.path.join("c:\\", "path")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv"):
                    with open(file, 'r') as f:
                        # Placeholder for calculations or processing on each CSV file.
                        pass

    # Concatenate all DataFrames in df_list and return along with the week list (w_list).
    return pd.concat(df_list), w_list

def get_lineage_class(metadata, id_list):
    """
    This function extracts the Pango lineage names for a given list of IDs from the metadata.

    Parameters:
    - metadata: DataFrame containing variant data, including Pango lineage and Accession IDs.
    - id_list: List of Accession IDs for which the lineage names are to be extracted.

    Returns:
    - A list of Pango lineage names corresponding to each ID in the id_list.
    """

    # Initialize an empty list to store the lineage names for each variant.
    variant_name_list = []

    # Loop through each ID in the provided id_list.
    for id in id_list:
        # For each ID, perform the following steps:
        # 1. Filter the metadata DataFrame to find the row where 'Accession.ID' matches the current ID.
        # 2. Extract the 'Pango.lineage' value from this filtered row.
        # 3. Add the extracted lineage name to the variant_name_list.
        # Note: 'values[0]' is used to get the first item from the resulting array, assuming each ID is unique.
        variant_name_list.append(metadata[metadata['Accession.ID'] == id]['Pango.lineage'].values[0])

    # Return the compiled list of lineage names.
    return variant_name_list

def map_lineage_to_finalclass(class_list, non_neutral):
    """
    This function maps a list of lineages to a final class based on their neutrality.
    Parameters:
    - class_list: A list of lineages to be classified.
    - non_neutral: A list of lineages that are considered non-neutral.

    Returns:
    - final_class_list: A list where each lineage from class_list is classified as -1 (non-neutral) or 1 (neutral).
    """

    # Initialize an empty list to store the final classification of each lineage.
    final_class_list = []

    # Iterate over each lineage in the class_list.
    for c in class_list:
        # Check if the current lineage is in the list of non-neutral lineages.
        if c in non_neutral:
            # If it is non-neutral, append -1 to the final_class_list.
            final_class_list.append(-1)
        else:
            # If it is not non-neutral (i.e., it is neutral), append 1 to the final_class_list.
            final_class_list.append(1)

    # Return the list containing the final classification for each lineage.
    return final_class_list

def autoencoder_training_GPU(autoencoder, train1, train2, nb_epoch, batch_size):
    """
    This function trains an autoencoder model, utilizing a GPU if available.
    Parameters:
    - autoencoder: The autoencoder model to be trained.
    - train1: The input data for training.
    - train2: The target data for training (can be the same as train1 for autoencoders).
    - nb_epoch: The number of epochs for training.
    - batch_size: The size of batches used in training.

    Returns:
    - history: A history object containing the training progress information.
    """
    k = 2.5
    # Check if a GPU is available in the TensorFlow environment.
    if tf.config.experimental.list_physical_devices('GPU'):
        # If a GPU is available, configure TensorFlow to use it.
        with tf.device('/GPU:0'):
            # Train the autoencoder on the GPU using the specified training data, epochs, and batch size.
            # 'shuffle=True' shuffles the training data, and 'verbose=1' prints out the training progress.
            history = autoencoder.fit(train1, train2,
                                      epochs=nb_epoch,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      verbose=1
                                      ).history
    else:
        # If no GPU is available, print a message and use the CPU for training.
        print("No GPU available. Using CPU instead.")
        history = autoencoder.fit(train1, train2,
                                  epochs=nb_epoch,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  verbose=1
                                  ).history

    # Return the history object capturing the training progress.
    return history,k

def weeks_before(summary, lineage_of_interest):
    """
    This function calculates the number of weeks before each lineage in the summary was first predicted
    as an anomaly compared to when it was first recognized.

    Parameters:
    - summary: A list of lists containing lineages and related information.

    Returns:
    - A list of differences in weeks for each lineage between its first prediction as an anomaly and its recognition.
    """

    # A predefined NumPy array containing lineages and their respective weeks of recognition.
    week_identification_np=np.array([['B.1',13],['B.1.1 ',22],['B.1.159',24],['B.1.416',28],['B.1.160',28],['B.1.367',34],['B.1.177',41],['B.1.1.7',55],['B.1.617.2',76], ['AY.43',83],['AY.42',83],['AY.98.1',84],['AY.122',94],['AY.125',94],['AY.5',99],['AY.4',99],['BA.1',110],['BA.2',112],['BA.2.12.1',130],['BA.5',138],['BA.2.9',142],['BA.2.75',144],['CH.1.1',159],['XBB.1.5',168],['XBB.1.16',188],['BE.6',148],['BF.7',148],['BQ.1.1',152],['BN.1',161]])
    week_growing_np=np.array([['B.1',5],['B.1.1 ',13],['B.1.159',14],['B.1.416',1],['B.1.160',26],['B.1.367',3],['B.1.177',23],['B.1.1.7',10],['B.1.617.2',9], ['AY.43',13],['AY.42',9],['AY.98.1',12],['AY.122',24],['AY.125',16],['AY.5',25],['AY.4',24],['BA.1',7],['BA.2',7],['BA.2.12.1',7],['BA.5',7],['BA.2.9',34],['BA.2.75',8],['CH.1.1',7],['XBB.1.5',2],['XBB.1.16',12],['BE.6',6],['BF.7',13],['BQ.1.1',7],['BN.1',14]])

    # Convert the input summary to a NumPy array for easier processing.
    summary_np = np.array(summary)

    # Extract lineages from the summary array.
    Lineages = summary_np[:, 0]

    # Create a dictionary with counts of each lineage.
    Lineages_dict = Counter(Lineages)

    # Initialize a list to store the final output.
    final_distance = []

    # Iterate through each unique lineage in the dictionary.
    for k in Lineages_dict.keys():
        if k not in lineage_of_interest:
            continue  # Skip if the lineage not in lineage of interest'

        # Find indices in summary where the current lineage is present.
        i_k = np.where(summary_np == k)[0]

        # Find the index in the predefined array for the current lineage.
        i_w = np.where(week_identification_np == k)[0]

        # Find the index in the predefined array for the current lineage.
        i_f = np.where(week_growing_np == k)[0]

        # Extract recognized weeks for the current lineage.
        week_recognize = np.array(list(map(int, week_identification_np[i_w, 1])))
        interval_10_percent = np.array(list(map(int, week_growing_np[i_f, 1])))
        # Extract predicted counts and anomaly weeks for the current lineage from the summary.
        predicted = np.array(list(map(int, summary_np[i_k, 2])))
        week_an = np.array(list(map(int, summary_np[i_k, 3])))

        # Determine the first week when an anomaly was predicted.
        Index_first_prediction = np.where(predicted > 0)[0]
        if len(Index_first_prediction) == 0:
            continue  # Skip if there's no prediction
        week_first_prediction = min(list(week_an[Index_first_prediction]))
        week_first_prediction_true = week_first_prediction + 1

        # Calculate the difference in weeks between recognized and first predicted anomaly week.
        week_before = np.array(week_recognize - week_first_prediction_true)
        fraction_before = np.array((week_recognize - week_first_prediction_true)/interval_10_percent)

        # Append the result to the final_distance list.
        summary = [k, week_before,fraction_before]
        final_distance.append(summary)

    # Return the list of differences for each lineage.
    return final_distance

def find_lineage_per_week(dataset, week, dictionary_lineage_weeks):
    """
    This function filters a dataset to retrieve rows corresponding to specific lineages for a given week.

    Parameters:
    - dataset: The dataset to be filtered, assumed to be a NumPy array.
    - week: The specific week for which lineages are to be filtered.
    - dictionary_lineage_weeks: A dictionary mapping weeks to lineages.

    Returns:
    - results: A NumPy array containing the filtered dataset rows.
    """

    # Assume that the lineage column is the last column in the dataset.
    lineage_column = dataset.shape[1] - 1

    # Extract the lineages for the specified week from the dictionary.
    weekly_lineages = dictionary_lineage_weeks[week]

    # Create an empty NumPy array to store the results.
    results = np.empty((0, dataset.shape[1]), dtype=dataset.dtype)

    # Iterate through the dataset and select only the rows with lineages that correspond to the specified week.
    for lineage in weekly_lineages:
        # Find the rows in the dataset where the lineage matches.
        selected_rows = dataset[np.where(dataset[:, lineage_column] == lineage)]

        # Stack the selected rows onto the results array.
        results = np.vstack((results, selected_rows))

    # Return the array containing the selected rows.
    return results

def find_lineage_per_week(dataset, week, dictionary_lineage_weeks):
    """
    This function filters a dataset to retrieve rows corresponding to specific lineages for a given week.

    Parameters:
    - dataset: The dataset to be filtered, assumed to be a NumPy array.
    - week: The specific week for which lineages are to be filtered.
    - dictionary_lineage_weeks: A dictionary mapping weeks to lineages.

    Returns:
    - results: A NumPy array containing the filtered dataset rows.
    """

    # Assume that the lineage column is the last column in the dataset.
    lineage_column = dataset.shape[1] - 1

    # Extract the lineages for the specified week from the dictionary.
    weekly_lineages = dictionary_lineage_weeks[week]

    # Create an empty NumPy array to store the results.
    results = np.empty((0, dataset.shape[1]), dtype=dataset.dtype)

    # Iterate through the dataset and select only the rows with lineages that correspond to the specified week.
    for lineage in weekly_lineages:
        # Find the rows in the dataset where the lineage matches.
        selected_rows = dataset[np.where(dataset[:, lineage_column] == lineage)]

        # Stack the selected rows onto the results array.
        results = np.vstack((results, selected_rows))

    # Return the array containing the selected rows.
    return results

def find_indices_lineage_per_week(lineage_column, week, dictionary_lineage_weeks):
    """
    This function finds row indices in a dataset for specific lineages corresponding to a given week.

    Parameters:
    - lineage_column: The column in the dataset containing lineage information.
    - week: The specific week for which lineages are to be matched.
    - dictionary_lineage_weeks: A dictionary mapping weeks to lineages.

    Returns:
    - indices_rows_np: A NumPy array of integers containing the indices of rows matching the specified lineages for the week.
    """

    # Extract the lineages for the specified week from the dictionary.
    weekly_lineages = dictionary_lineage_weeks[week]

    # Create an empty list to store the indices of corresponding rows.
    indices_rows = []

    # Iterate through the lineage column and select only the indices of rows with lineages corresponding to the specified week.
    for i, lineage in enumerate(lineage_column):
        if lineage in weekly_lineages:
            indices_rows.append(i)

    # Convert the indices list to a NumPy array of integers.
    indices_rows_np = np.array(indices_rows, dtype=int)

    # Return the array of indices.
    return indices_rows_np

def find_indices_lineage_per_week_sublineage(lineage_column, week, dictionary_lineage_weeks):
    """
    This function finds row indices in a dataset for specific lineages and their sublineages corresponding to a given week,
    excluding the sublineages that are sublineages of new lineages in subsequent weeks.

    Parameters:
    - lineage_column: The column in the dataset containing lineage information.
    - week: The specific week for which lineages are to be matched.
    - dictionary_lineage_weeks: A dictionary mapping weeks to lineages.

    Returns:
    - indices_rows_np: A NumPy array of integers containing the indices of rows matching the specified lineages and their sublineages for the week,
                       excluding those sublineages of new lineages in subsequent weeks.
    """
    # Extract the lineages for the specified week from the dictionary.
    weekly_lineages = set(dictionary_lineage_weeks[week])

    # Extract lineages for the previous weeks.
    previous_lineages = set()
    weeks = list(dictionary_lineage_weeks.keys())
    current_week_index = weeks.index(week)
    previous_weeks = weeks[:current_week_index]

    for previous_week in previous_weeks:
        previous_lineages.update(dictionary_lineage_weeks[previous_week])

    # Extract lineages for the subsequent weeks.
    new_subsequent_lineages = set()
    subsequent_weeks = weeks[current_week_index + 1:]

    for subsequent_week in subsequent_weeks:
        for lineage in dictionary_lineage_weeks[subsequent_week]:
            if lineage not in weekly_lineages:
                new_subsequent_lineages.add(lineage)

    # Create a set to store the indices of corresponding rows.
    indices_rows = []

    # Iterate through the lineage column and find sublineages for the specified week.
    for i, lineage in enumerate(lineage_column): # prima c'era or
        if any(lineage == weekly_lineage or lineage.startswith(weekly_lineage + '.') for weekly_lineage in
               weekly_lineages):
            # Check if this lineage is a sublineage of any new subsequent lineages.
            if not any(lineage == new_lineage or lineage.startswith(new_lineage + '.') for new_lineage in new_subsequent_lineages):
                indices_rows.append(i)

    # Convert the indices list to a NumPy array of integers.
    indices_rows_np = np.array(indices_rows, dtype=int)

    # Return the array of indices.
    return indices_rows_np


def test_normality(autoencoder, train_model):
    """
    This function tests the normality of the mean squared error (MSE) distribution of autoencoder predictions
    on training data using the Shapiro-Wilk test.

    Parameters:
    - autoencoder: The autoencoder model used for making predictions.
    - train_model: The training data on which the autoencoder is tested.

    Returns:
    - p_value: The p-value from the Shapiro-Wilk test indicating normality of the MSE distribution.
               A value of -1 indicates insufficient data for the test (less than 3 data points).
    - mse: The calculated mean squared errors for each point in the training data.
    """

    # Calculates autoencoder predictions on training data.
    predictions = autoencoder.predict(train_model)

    # Calculates the mean squared error (MSE) of the autoencoder on the training data.
    # This is done by taking the mean of the squared differences between actual and predicted values.
    mse = np.mean(np.power(train_model - predictions, 2), axis=1)

    # Test the normality of the MSE using the Shapiro-Wilk test.
    # Shapiro-Wilk test requires more than 3 data points, so check the length of mse.
    if len(mse) > 3:
        # If there are sufficient data points, perform the Shapiro-Wilk test.
        # Flatten the mse array for the test and store the p-value.
        _, p_value = shapiro(mse.flatten())
    else:
        # If there are not enough data points, set p_value to -1.
        p_value = -1

    # Return the p-value and the calculated MSE.
    return p_value, mse

def model(input_dim, encoding_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, hidden_dim_4, hidden_dim_5, reduction_factor, path_salvataggio_file):
    """
    This function creates and returns an autoencoder model.
    Parameters:
    - input_dim: Dimension of the input layer.
    - encoding_dim: Dimension of the encoding layer.
    - hidden_dim_1 to hidden_dim_5: Dimensions of the hidden layers.
    - reduction_factor: Factor used for L2 regularization to reduce overfitting.
    - path_salvataggio_file: Path where the model checkpoint will be saved.

    Returns:
    - autoencoder: The constructed TensorFlow Keras autoencoder model.
    """

    # Input Layer
    # Create the input layer with the specified dimension.
    input_layer = tf.keras.layers.Input(shape=(input_dim,))

    # Encoder
    # Construct the encoder part of the autoencoder using dense layers, dropout for regularization,
    # and various activation functions.
    encoder = tf.keras.layers.Dense(encoding_dim, activation="tanh", activity_regularizer=tf.keras.regularizers.l2(reduction_factor))(input_layer)
    encoder = tf.keras.layers.Dropout(0.3)(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_2, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_3, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_4, activation='relu')(encoder)

    # Central Layer with Noise
    # Add a latent layer with leaky ReLU activation and introduce noise to make the model more robust.
    latent = tf.keras.layers.Dense(hidden_dim_5, activation=tf.nn.leaky_relu)(encoder)
    noise_factor = 0.1
    latent_with_noise = tf.keras.layers.Lambda(
        lambda x: x + noise_factor * tf.keras.backend.random_normal(shape=tf.shape(x)))(latent)

    # Decoder
    # Build the decoder part of the autoencoder to reconstruct the input from the encoded representation.
    decoder = tf.keras.layers.Dense(hidden_dim_4, activation='relu')(latent_with_noise)
    decoder = tf.keras.layers.Dense(hidden_dim_3, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(hidden_dim_2, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(decoder)
    decoder = tf.keras.layers.Dropout(0.3)(decoder)
    decoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(decoder)

    # Output Layer
    # Define the output layer to reconstruct the original input.
    decoder = tf.keras.layers.Dense(input_dim, activation='tanh', activity_regularizer=tf.keras.regularizers.l2(reduction_factor))(decoder)

    # Autoencoder Model
    # Define the autoencoder model that maps the input to its reconstruction.
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
    autoencoder.summary()

    # Callbacks for Model Checkpoint and Early Stopping
    # Set up a checkpoint to save the model and early stopping to prevent overfitting.
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=path_salvataggio_file + "/autoencoder_fraud_AERNS.h5",
                                            mode='min', monitor='loss', verbose=2, save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.001,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True)

    # Compile the Autoencoder
    # Use mean squared error as the loss function and Adam optimizer for training.
    autoencoder.compile(metrics=['mse'],
                        loss='mean_squared_error',
                        optimizer='adam')

    # Return the compiled autoencoder model.
    return autoencoder


def kmers_importance(prediction, true_sequence, kmers):
    """
    This function identifies the most important k-mers based on the differences between predicted and true sequences.

    Parameters:
    - prediction: A list of predicted sequence values.
    - true_sequence: A list of actual sequence values.
    - kmers: A list of k-mers.

    Returns:
    - kmers_selected: A list of the top 6 k-mers based on their importance.
    """

    # Initialize a list to store the differences between predicted and true sequence values.
    differences = []

    # Iterate over the sequences and compute the difference between each predicted and true value.
    for i in range(len(prediction)):
        seq_pred = prediction[i]
        seq_real = true_sequence[i]
        differences.append(seq_pred - seq_real)

    # Sort the indices based on the differences, in descending order (largest differences first).
    sorted_indices = sorted(range(len(differences)), key=lambda k: differences[k], reverse=True)

    # Select the indices corresponding to the top 6 differences.
    #top_6_indices = sorted_indices[:6]

    # Retrieve the k-mers corresponding to these top 6 indices.
    kmers_selected = [kmers[i] for i in sorted_indices]

    # Return the selected k-mers.
    return kmers_selected


def selection_kmers(AE_prediction, True_sequences, kmers, AE_classes, identifier, output_filename="summary_KMERS.csv"):
    """
    This function selects k-mers based on anomalies detected by an autoencoder model and outputs the results to a CSV file.
    The function exits if no anomalies are detected.

    Inputs:
    - AE_prediction: List of sequences predicted by the model.
    - True_sequences: List of real sequences encoded in the k-mers space.
    - kmers: List of filtered k-mers that differ from zero by at least 0.01%.
    - AE_classes: Numpy array of classes defined by the model (1 for normal, -1 for anomaly).
    - identifier: List of sequence IDs.
    - output_filename: String specifying the path and name of the output CSV file.

    Output:
    - A CSV file with the following columns:
        1) The first column contains the sequence ID.
        2) The other columns contain the selected k-mers.
    """

    # Identify indices where the model has classified the sequences as anomalies.
    anomaly_indices = np.where(AE_classes == -1)[0]

    # Exit the function if no anomalies are detected.
    if len(anomaly_indices) == 0:
        print("No anomalies detected. Exiting function.")
        return

    # Extract predictions, true sequences, and identifiers for the anomalies.
    AE_prediction_anomalies = [AE_prediction[i] for i in anomaly_indices]
    True_sequences_anomalies = [True_sequences[i] for i in anomaly_indices]
    identificativo_anomalies = [identifier[i] for i in anomaly_indices]

    # Initialize a list to store the summary data.
    summary = []

    # Iterate over each anomaly to select k-mers and compile the summary information.
    for i in range(len(AE_prediction_anomalies)):
        prediction = AE_prediction_anomalies[i]
        real = True_sequences_anomalies[i]

        # Determine the importance of k-mers based on the prediction and real sequence.
        kmers_selected = kmers_importance(prediction, real, kmers)

        # Append the sequence identifier and the selected k-mers to the summary.
        summary.append([identificativo_anomalies[i]] + kmers_selected)

    # Write the summary to a CSV file.
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write the header of the CSV file.
        header = ['Identificativo'] + ['kmer_' + str(i + 1) for i in range(len(summary[0]) - 1)]
        csvwriter.writerow(header)

        # Write each row of the summary to the CSV file.
        csvwriter.writerows(summary)

def lookup(y_test_i_predict, y_test_step_i, knowledge, mse):
    """
    This function modifies predictions and Mean Squared Error (MSE) values based on prior knowledge of lineages,
    making it robust to spaces in lineage names.

    Parameters:
    - y_test_i_predict: A list/array of initial predictions, where -1 and 1 represent different classes.
    - y_test_step_i: A list/array of lineages corresponding to each prediction.
    - knowledge: A list of lineages that are known or considered significant.
    - mse: A list/array of Mean Squared Error (MSE) values corresponding to each prediction.

    Returns:
    - prediction: The modified list of predictions.
    - mse: The modified list of Mean Squared Error (MSE) values.
    """

    # Clean up spaces in the lineages and known lineages lists
    lineages = [lineage.strip() for lineage in y_test_step_i]
    lineages_known = [known.strip() for known in knowledge]

    # Assign the initial predictions to a variable.
    prediction = y_test_i_predict

    # Assign the MSE values to a variable.
    mse = mse

    # Iterate through each lineage in the list of lineages.
    for i, lineage in enumerate(lineages):
        # Check if the lineage is in the list of known lineages and if the corresponding prediction is -1.
        if lineage in lineages_known and prediction[i] == -1:
            # If both conditions are met, modify the prediction for this lineage to 1.
            prediction[i] = 1
            # Also, set the corresponding MSE value to 0.
            mse[i] = 0

    # Return the modified list of predictions and MSE values.
    return prediction, mse

def lookup_post(y_test_i_predict, y_test_step_i, knowledge):
    """
    This function modifies predictions based on prior knowledge of lineages,
    making it robust to spaces in lineage names.

    Parameters:
    - y_test_i_predict: A list/array of initial predictions, where -1 and 1 represent different classes.
    - y_test_step_i: A list/array of lineages corresponding to each prediction.
    - knowledge: A list of lineages that are known or considered important.

    Returns:
    - prediction: The modified list of predictions.
    """

    # Clean up spaces in the lineages and known lineages lists
    lineages = [lineage.strip() for lineage in y_test_step_i]
    lineages_known = [known.strip() for known in knowledge]

    # Assign the initial predictions to a variable.
    prediction = y_test_i_predict

    # Iterate through each lineage in the list of lineages.
    for i, lineage in enumerate(lineages):
        # Check if the lineage is in the list of known lineages and if the corresponding prediction is -1.
        if lineage in lineages_known and prediction[i] == -1:
            # If both conditions are met, modify the prediction for this lineage to 1.
            prediction[i] = 1

    # Return the modified list of predictions.
    return prediction

def lineages_of_interest():
    ## Valid Lineages
    valid_lineage_FDLs = ['B.1', 'B.1.1','B.1.159','B.1.416', 'B.1.160', 'B.1.367', 'B.1.177', 'AY.43', 'B.1.1.7', 'AY.4','AY.5','AY.122','AY.125','B.1.617.2',
                     'AY.98.1','AY.42', 'BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1','BA.2.75']  # mettere i lineage che definisco come classe# mettere i lineage che definisco come classe

    valid_lineage_newFDLS = ['BA.5','BE.6','BF.7','BN.1','BQ.1.1','CH.1.1','XBB.1.5','XBB.1.16']
    valid_lineage = valid_lineage_FDLs + valid_lineage_newFDLS

    # Valid Lineages PRC
    valid_lineage_prc = [
        ['B.1', 'B.1.1','B.1.159','B.1.416', 'B.1.160', 'B.1.367', 'B.1.177', 'AY.43', 'B.1.1.7', 'AY.4','AY.5','AY.122','AY.125','B.1.617.2','AY.98.1','AY.42', 'BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1','BA.2.75','BA.5','BE.6','BF.7','BN.1','BQ.1.1','CH.1.1','XBB.1.5','XBB.1.16'],# Start
        ['B.1.1','B.1.159','B.1.416', 'B.1.160', 'B.1.367', 'B.1.177', 'AY.43', 'B.1.1.7', 'AY.4','AY.5','AY.122','AY.125','B.1.617.2','AY.98.1','AY.42', 'BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1','BA.2.75','BA.5','BE.6','BF.7','BN.1','BQ.1.1', 'CH.1.1','XBB.1.5', 'XBB.1.16'],#13-22
        ['B.1.159', 'B.1.416', 'B.1.160', 'B.1.367', 'B.1.177', 'AY.43', 'B.1.1.7', 'AY.4', 'AY.5', 'AY.122','AY.125', 'B.1.617.2', 'AY.98.1', 'AY.42', 'BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1', 'BA.2.75', 'BA.5', 'BE.6','BF.7', 'BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'],  # 22-24
        ['B.1.416', 'B.1.160', 'B.1.367', 'B.1.177', 'AY.43', 'B.1.1.7', 'AY.4', 'AY.5', 'AY.122', 'AY.125','B.1.617.2', 'AY.98.1', 'AY.42', 'BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1', 'BA.2.75', 'BA.5', 'BE.6', 'BF.7','BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'], #24 -28
        ['B.1.367', 'B.1.177', 'AY.43', 'B.1.1.7', 'AY.4', 'AY.5', 'AY.122', 'AY.125','B.1.617.2', 'AY.98.1', 'AY.42', 'BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1', 'BA.2.75', 'BA.5', 'BE.6', 'BF.7','BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'], # 28-34
        ['B.1.177', 'AY.43', 'B.1.1.7', 'AY.4', 'AY.5', 'AY.122', 'AY.125','B.1.617.2', 'AY.98.1', 'AY.42', 'BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1', 'BA.2.75', 'BA.5', 'BE.6', 'BF.7','BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'],# 34-41
        ['AY.43', 'B.1.1.7', 'AY.4', 'AY.5', 'AY.122', 'AY.125','B.1.617.2', 'AY.98.1', 'AY.42', 'BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1', 'BA.2.75', 'BA.5', 'BE.6', 'BF.7','BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'],# 41-55
        ['AY.43', 'AY.4', 'AY.5', 'AY.122', 'AY.125','B.1.617.2', 'AY.98.1', 'AY.42', 'BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1', 'BA.2.75', 'BA.5', 'BE.6', 'BF.7','BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'],# 55-76
        ['AY.43', 'AY.4', 'AY.5', 'AY.122', 'AY.125', 'AY.98.1', 'AY.42', 'BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1', 'BA.2.75', 'BA.5', 'BE.6', 'BF.7','BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'],# 76-82
        ['AY.4', 'AY.5', 'AY.122', 'AY.125', 'AY.98.1', 'BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1', 'BA.2.75', 'BA.5', 'BE.6', 'BF.7','BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'],# 82-84
        ['AY.4', 'AY.5', 'AY.122', 'AY.125', 'BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1', 'BA.2.75', 'BA.5', 'BE.6', 'BF.7','BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'],# 84-94
        ['AY.4', 'AY.5', 'BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1', 'BA.2.75', 'BA.5', 'BE.6', 'BF.7','BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'], #94-100
        ['BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1', 'BA.2.75', 'BA.5', 'BE.6', 'BF.7','BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'],#100-110
        ['BA.2', 'BA.2.9', 'BA.2.12.1', 'BA.2.75', 'BA.5', 'BE.6', 'BF.7','BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'],#110-112
        ['BA.2.9', 'BA.2.12.1', 'BA.2.75', 'BA.5', 'BE.6', 'BF.7','BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'],#112-130
        ['BA.2.9', 'BA.2.75', 'BA.5', 'BE.6', 'BF.7','BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'],#130-138
        ['BA.2.9', 'BA.2.75', 'BE.6', 'BF.7','BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'],#138-142
        ['BA.2.75', 'BE.6', 'BF.7', 'BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'],#142-144
        ['BE.6', 'BF.7', 'BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'],#144-148
        ['BN.1', 'BQ.1.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'],# 148-152
        ['BN.1', 'CH.1.1', 'XBB.1.5', 'XBB.1.16'], #152-159
        ['BN.1', 'XBB.1.5', 'XBB.1.16'],#159-161
        ['XBB.1.5','XBB.1.16'],#161-168
        ['XBB.1.16'],#168-188
        []]

    dictionary_lineage_week = {
        13: ['B.1'],
        22: ['B.1', 'B.1.1'],
        24: ['B.1', 'B.1.1', 'B.1.159'],
        28: ['B.1', 'B.1.1', 'B.1.159', 'B.1.160', 'B.1.416'],
        34: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367'],
        41: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177'],
        55: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7'],
        76: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2'],
        82: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
             'AY.43', 'AY.42'],
        84: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
             'AY.43', 'AY.42', 'AY.98.1'],
        94: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
             'AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125'],
        100: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
              'AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4'],
        110: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
              'AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1'],
        112: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
              'AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2'],
        130: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
              'AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1'],
        138: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
              'AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5'],
        142: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
              'AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5',
              'BA.2.9'],
        144: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
              'AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5',
              'BA.2.9', 'BA.2.75'],
        148: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
              'AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5',
              'BA.2.9', 'BA.2.75', 'BE.6', 'BF.7'],
        152: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
              'AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5',
              'BA.2.9', 'BA.2.75', 'BE.6', 'BF.7', 'BQ.1.1'],
        159: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
              'AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5',
              'BA.2.9', 'BA.2.75', 'BE.6', 'BF.7', 'BQ.1.1', 'CH.1.1'],
        161: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
              'AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5',
              'BA.2.9', 'BA.2.75', 'BE.6', 'BF.7', 'BQ.1.1', 'CH.1.1', 'BN.1'],
        168: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
              'AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5',
              'BA.2.9', 'BA.2.75', 'BE.6', 'BF.7', 'BQ.1.1', 'CH.1.1', 'BN.1', 'XBB.1.5'],
        188: ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2',
              'AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5',
              'BA.2.9', 'BA.2.75', 'BE.6', 'BF.7', 'BQ.1.1', 'CH.1.1', 'BN.1', 'XBB.1.5', 'XBB.1.16']
    }

    lineage_know = [
                    [],
                    ['B.1'],
                    ['B.1', 'B.1.1'],
                    ['B.1', 'B.1.1','B.1.159'],
                    ['B.1', 'B.1.1','B.1.159','B.1.160','B.1.416'],
                    ['B.1', 'B.1.1', 'B.1.160','B.1.159','B.1.416','B.1.367'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367','B.1.177'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367','B.1.177','B.1.1.7'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367','B.1.177','B.1.1.7','B.1.617.2'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367','B.1.177','B.1.1.7','B.1.617.2','AY.43','AY.42'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367','B.1.177','B.1.1.7','B.1.617.2','AY.43','AY.42','AY.98.1'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367','B.1.177','B.1.1.7','B.1.617.2','AY.43','AY.42','AY.98.1','AY.122','AY.125'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367','B.1.177','B.1.1.7','B.1.617.2','AY.43','AY.42','AY.98.1','AY.122','AY.125','AY.5','AY.4'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367','B.1.177','B.1.1.7','B.1.617.2','AY.43','AY.42','AY.98.1','AY.122','AY.125','AY.5','AY.4','BA.1'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367','B.1.177','B.1.1.7','B.1.617.2','AY.43','AY.42','AY.98.1','AY.122','AY.125','AY.5','AY.4','BA.1','BA.2'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367','B.1.177','B.1.1.7','B.1.617.2','AY.43','AY.42','AY.98.1','AY.122','AY.125','AY.5','AY.4','BA.1','BA.2','BA.2.12.1'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367','B.1.177','B.1.1.7','B.1.617.2','AY.43','AY.42','AY.98.1','AY.122','AY.125','AY.5','AY.4','BA.1','BA.2','BA.2.12.1','BA.5'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367','B.1.177','B.1.1.7','B.1.617.2','AY.43','AY.42','AY.98.1','AY.122','AY.125','AY.5','AY.4','BA.1','BA.2','BA.2.12.1','BA.5', 'BA.2.9'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2','AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5','BA.2.9','BA.2.75'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2','AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5','BA.2.9','BA.2.75','BE.6','BF.7'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2','AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5','BA.2.9','BA.2.75','BE.6','BF.7', 'BQ.1.1'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2','AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5','BA.2.9','BA.2.75','BE.6','BF.7', 'BQ.1.1','CH.1.1'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2','AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5','BA.2.9','BA.2.75','BE.6','BF.7', 'BQ.1.1','CH.1.1','BN.1'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2','AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5','BA.2.9','BA.2.75','BE.6','BF.7', 'BQ.1.1','CH.1.1','BN.1','XBB.1.5'],
                    ['B.1', 'B.1.1', 'B.1.160', 'B.1.159', 'B.1.416', 'B.1.367', 'B.1.177', 'B.1.1.7', 'B.1.617.2','AY.43', 'AY.42', 'AY.98.1', 'AY.122', 'AY.125', 'AY.5', 'AY.4', 'BA.1', 'BA.2', 'BA.2.12.1', 'BA.5','BA.2.9','BA.2.75','BE.6','BF.7', 'BQ.1.1','CH.1.1','BN.1','XBB.1.5','XBB.1.16']
                    ]
    return valid_lineage,valid_lineage_prc,dictionary_lineage_week,lineage_know

def retraining_weeks():
    retraining_week = [13, 22, 24, 28, 34, 41, 55, 76, 82, 84, 94, 100, 110, 112, 130, 138, 142, 144, 148, 152, 159, 161, 168, 188]
    retraining_week_false_positive = [13, 22, 24, 28, 34, 41, 55, 76, 82, 84, 94, 100, 110, 112, 130, 138, 142, 144, 148, 152, 159, 161, 168, 188, 189]
    return retraining_week,retraining_week_false_positive

def write_feature(feature_model, path_to_save, name_txt):

    # Function to transform and write the content to the output file
    def write_feat(feature,path_to_save,name_file):
        with open(path_to_save+name_file, "w") as file:
            for elemento in feature:
                file.write(str(elemento) + "\n")

    def transform_and_write_content(input_content, output_path):
        with open(output_path, 'w') as file:
            for line in input_content:
                # Remove single quotes and split by comma and space
                elements = line.replace("'", "").split(', ')
                # Write each element on a new line in the output file
                for element in elements:
                    file.write(element + '\n')

    write_feat(feature_model,path_to_save,name_txt)
    # The path to the input file, which contains the original data
    # = path_to_save + name_txt # Replace with your input file path

    # The path to the output file, which will contain the transformed data
    #output_file_path = path_to_save + name_txt  # Replace with your desired output file path

    # Read the original content from the input file
    #with open(input_file_path, 'r') as file:
        #original_content = file.readlines()

    # Perform the transformation and write to the output file
    #transform_and_write_content(original_content, output_file_path)

def count_true_and_false_positives_top100(predicted_lineages, known_lineages):
    """
    Counts the true positives and false positives in a list of predicted lineages,
    and returns an associated list indicating positions as 1 for true positive and -1 for false positive.

    A true positive is a lineage that is either present in the list of known lineages or is a sublineage of a known element.
    A false positive is a lineage that does not meet the criteria to be a true positive.

    :param predicted_lineages: List of predicted lineages.
    :param known_lineages: List of known lineages.
    :return: A tuple (true_positives, false_positives, tp_fp_positions).
    """
    true_positives = 0
    false_positives = 0
    tp_fp_positions = []

    # Clean up spaces in the lineages lists
    cleaned_predicted_lineages = [lineage.strip() for lineage in predicted_lineages]
    cleaned_known_lineages = [lineage.strip() for lineage in known_lineages]

    for predicted in cleaned_predicted_lineages:
        if any(predicted.startswith(known) for known in cleaned_known_lineages):
            true_positives += 1
            tp_fp_positions.append(1)
        else:
            false_positives += 1
            tp_fp_positions.append(-1) # qua ci va meno 1

    return true_positives, false_positives, np.array(tp_fp_positions)

def count_true_and_false_positives_overall(predicted_lineages, known_lineages):
    """
    Counts the true positives and false positives in a list of predicted lineages,
    making it robust to spaces in lineage names.

    A true positive is a lineage that is either present in the list of known lineages or is a sublineage of a known element.
    A false positive is a lineage that does not meet the criteria to be a true positive.

    :param predicted_lineages: List of predicted lineages.
    :param known_lineages: List of known lineages.
    :return: A tuple (true_positives, false_positives).
    """
    true_positives = 0
    false_positives = 0

    # Clean up spaces in the lineages lists
    cleaned_predicted_lineages = [lineage.strip() for lineage in predicted_lineages]
    cleaned_known_lineages = [lineage.strip() for lineage in known_lineages]

    for predicted in cleaned_predicted_lineages:
        if any(predicted.startswith(known) for known in cleaned_known_lineages):
            true_positives += 1
        else:
            false_positives += 1

    return true_positives, false_positives


# def plot_weekly_precision(precisions, file_path,title):
#     """
#     This function takes a list of precision values for each week and creates a line plot using Seaborn.
#
#     :param precisions: List of precision values (float or int) for each week.
#     """
#     # Create a DataFrame with the precision values
#     data = pd.DataFrame({'Week': range(len(precisions)), 'Precision': precisions})
#
#     # Create a line plot
#     sns.barplot(x='Week', y='Precision', data=data)
#
#     # Add titles and labels
#     plt.title('Weekly Precision')
#     plt.xlabel('Week')
#     plt.ylabel('Precision')
#
#     # Save the plot
#     plt.savefig(file_path+title)


def plot_weekly_precision(precisions, file_path, title):
    """
    This function takes a list of precision values for each week and creates a line plot using Matplotlib.

    :param precisions: List of precision values (float or int) for each week.
    :param file_path: File path to save the plot.
    :param title: Title for the plot.
    """
    # Create a DataFrame with the precision values
    data = pd.DataFrame({'Week': range(1, len(precisions) + 1), 'Precision': precisions})

    # Create a line plot
    plt.figure(figsize=(10, 6))
    plt.plot('Week', 'Precision', data=data, marker='o')

    # Add titles and labels
    plt.title(title)
    plt.xlabel('Week')
    plt.ylabel('Precision')

    # Save the plot
    plt.savefig(file_path + title + '.png')
    plt.close()

    return file_path + title + '.png'


def true_lineages_week(test_set, true_positives):
    """
    Counts the number of true positive lineages present in a test set,
    making it robust against whitespace variations.

    Args:
    test_set (list): List of lineages present in the test set.
    true_positives (list): List of lineages considered as true positives.

    Returns:
    int: Number of true positives in the test set.
    """
    count = 0
    # Trimming whitespace in both lists
    trimmed_test_set = [lineage.strip() for lineage in test_set]
    trimmed_true_positives = [lineage.strip() for lineage in true_positives]

    for lineage in trimmed_test_set:
        if lineage in trimmed_true_positives:
            count += 1

    return count

def covered_area(measure_sensitivity, lineage_of_interest):
    """
    Calculate the covered area for different variants over weeks based on sensitivity measurements.

    Parameters:
    - measure_sensitivity: A list of lists, where each inner list contains information about a variant,
                            its total count, predicted count, and the week number.

    Returns:
    - final_area: A list of lists, each containing the variant name and its calculated area.
    - only_area: A list of calculated areas for each variant.
    """
    final_area = []
    only_area = []
    measure_sensitivity_np = np.array(measure_sensitivity)
    Variants = measure_sensitivity_np[:, 0]
    Weeks = np.array(list(map(int, measure_sensitivity_np[:, 3])))
    total_official = np.array(list(map(int, measure_sensitivity_np[:, 1])))
    variant_dict = Counter(Variants)

    for k in variant_dict.keys():
        if k not in lineage_of_interest:
            continue
        indices_k = np.where(measure_sensitivity_np == k)[0] # Find indices where my variant is
        total = np.array(list(map(int, measure_sensitivity_np[indices_k, 1])))
        predicted = np.array(list(map(int, measure_sensitivity_np[indices_k, 2])))
        week_analysis = np.array(list(map(int, measure_sensitivity_np[indices_k, 3])))
        first_prediction_index = np.where(predicted > 0)[0] # Find index of first prediction
        if len(first_prediction_index) == 0:
            continue
        first_prediction_week = min(list(week_analysis[first_prediction_index]))
        index_first_prediction_week = np.where(week_analysis == first_prediction_week) # Find the index of the week and sum the second column
        total_lineage = total[index_first_prediction_week]
        all_weeks_indices = np.where(Weeks == first_prediction_week)
        total_seq_week = np.sum(total_official[all_weeks_indices])
        area = total_lineage / total_seq_week
        final_area.append([k, area])
        only_area.append(area)

    return final_area, only_area

def write_precision(precision,week):
    # First, we filter the precision values based on the condition.
    filtered_precision_paper = [precision[i] for i in range(len(week)) if precision[i] > 0]
    filtered_precision_week_1 = [precision[i] for i in range(len(week)) if week[i] > 1]
    filtered_precision_week_20 = [precision[i] for i in range(len(week)) if week[i] > 20]

    # Median
    median_precision_paper = np.mean(filtered_precision_paper)
    q1_prec_paper, q3_prec_paper = np.percentile(filtered_precision_paper, [25, 75])

    median_precision_week_1 = np.mean(filtered_precision_week_1)
    q1_prec_week_1, q3_prec_week_1 = np.percentile(filtered_precision_week_1, [25, 75])

    median_precision_week_20 = np.mean(filtered_precision_week_20)
    q1_prec_week_20, q3_prec_week_20 = np.percentile(filtered_precision_week_20, [25, 75])

    PPV = (median_precision_week_1 + median_precision_week_20) / 2
    q1 = (q1_prec_week_1 + q1_prec_week_20) / 2
    q3 = (q3_prec_week_1 + q3_prec_week_20) / 2
    return [
        ['The median is (paper) ' + str(median_precision_paper), 'The 25th percentile is (paper) ' + str(q1_prec_paper),
         'The 75th percentile is (paper) ' + str(q3_prec_paper)],
        ['The median is (>1) ' + str(median_precision_week_1), 'The 25th percentile is (>1) ' + str(q1_prec_week_1),
         'The 75th percentile is (>1) ' + str(q3_prec_week_1)],
        ['The median is (>n) ' + str(median_precision_week_20), 'The 25th percentile is (>n) ' + str(q1_prec_week_20),
         'The 75th percentile is (>n) ' + str(q3_prec_week_20)],
        ['The median is ' + str(PPV), 'The 25th percentile is ' + str(q1), 'The 75th percentile is ' + str(q3)]]