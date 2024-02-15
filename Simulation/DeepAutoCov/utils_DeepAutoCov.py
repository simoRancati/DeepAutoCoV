# Import the Library
import pandas as pd
import os
import tensorflow as tf
import numpy as np
from collections import Counter
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import csv


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
    return history


def weeks_before(summary):
    """
    This function calculates the number of weeks before each lineage in the summary was first predicted
    as an anomaly compared to when it was first recognized.

    Parameters:
    - summary: A list of lists containing lineages and related information.

    Returns:
    - A list of differences in weeks for each lineage between its first prediction as an anomaly and its recognition.
    """

    # A predefined NumPy array containing lineages and their respective weeks of recognition.
    week_identification_np = np.array([['Lineages', 10]])
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
        if k == 'unknown':
            continue  # Skip if the lineage is 'unknown'

        # Find indices in summary where the current lineage is present.
        i_k = np.where(summary_np == k)[0]

        # Find the index in the predefined array for the current lineage.
        i_w = np.where(week_identification_np == k)[0]

        # Extract recognized weeks for the current lineage.
        week_recognize = np.array(list(map(int, week_identification_np[i_w, 1])))

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

        # Append the result to the final_distance list.
        summary = [k, week_before]
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


def model(input_dim, encoding_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, hidden_dim_4, hidden_dim_5,
          reduction_factor, path_salvataggio_file):
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
    encoder = tf.keras.layers.Dense(encoding_dim, activation="tanh",
                                    activity_regularizer=tf.keras.regularizers.l2(reduction_factor))(input_layer)
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

    # costant egual 2.5
    th = 2.5

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
    decoder = tf.keras.layers.Dense(input_dim, activation='tanh',
                                    activity_regularizer=tf.keras.regularizers.l2(reduction_factor))(decoder)

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
    return autoencoder, th


def plot_sma(vector, window_size, path_save):
    """
    It calculates the Simple Moving Average (SMA) of a vector and plots it together with the barplot of the vector.
    The window_size parameter indicates the size of the moving window.
    """
    sma = np.convolve(vector, np.ones(window_size) / window_size, mode='valid')  # SMA

    fig, ax1 = plt.subplots(figsize=(20, 12))  # Create a figure and a subplot.

    # Plot the bar graph
    ax1.bar(range(len(vector)), vector, 0.4, color='#66c2a5', alpha=0.7)
    ax1.plot(range(window_size - 1, len(vector)), sma, 'r')
    ax1.set_title(str('False positive rate'), fontsize=26)
    # ax1.grid(False)  # Remove grid lines

    ax1.set_xlabel('Week', fontsize=24)  # Set x-axis label
    ax1.set_ylabel('False Positive Rate', fontsize=24)  # Set y-axis label
    ax1.tick_params(axis='both', which='major', labelsize=24)  # Set tick label size

    # Create an inset axes for the boxplot
    ax2 = inset_axes(ax1, width="40%", height="30%", loc='upper center')
    data_fp = {"False positive rate": vector}
    df_fp = pd.DataFrame(data_fp)
    sns.boxplot(x="False positive rate", data=df_fp, ax=ax2)
    ax2.set_xlabel("FPR", fontsize=22)
    ax2.tick_params(axis='x', labelsize=22)  # Increase x-axis label size for boxplot
    ax2.grid(False)  # Remove grid lines
    plt.savefig(str(path_save) + '/FalsePositiveRate.png', bbox_inches='tight')
    plt.show()


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
    top_6_indices = sorted_indices[:6]

    # Retrieve the k-mers corresponding to these top 6 indices.
    kmers_selected = [kmers[i] for i in top_6_indices]

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
    valid_lineage_FDLs = []  # Valid lineages that consider important to discover

    valid_lineage_newFDLS = []  # Add additional lineages (at user's discretion)
    valid_lineage = valid_lineage_FDLs + valid_lineage_newFDLS

    # Valid Lineages PRC
    valid_lineage_prc = [[]]  # For each week of retraining the valid lineages change

    dictionary_lineage_week = {
    }

    lineage_know = [
    ]  # Lineage that a certain instant (retraining weeks) become know by pubblic health authority (Database lineages know)
    return valid_lineage, valid_lineage_prc, dictionary_lineage_week, lineage_know


def retraining_weeks():
    retraining_week = []  # [weeks of retraining (third time when a lineage reach the 10 % of frequency)]
    retraining_week_false_positive = []  # [weeks of retraining (third time when a lineage reach the 10 % of frequency), the last week]
    return retraining_week, retraining_week_false_positive


def write_feature(feature_model, path_to_save, name_txt):
    # Function to transform and write the content to the output file
    def write_feat(feature, path_to_save, name_file):
        with open(path_to_save + name_file, "w") as file:
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

    write_feat(feature_model, path_to_save, name_txt)
    # The path to the input file, which contains the original data
    # = path_to_save + name_txt # Replace with your input file path

    # The path to the output file, which will contain the transformed data
    # output_file_path = path_to_save + name_txt  # Replace with your desired output file path

    # Read the original content from the input file
    # with open(input_file_path, 'r') as file:
    # original_content = file.readlines()

    # Perform the transformation and write to the output file
    # transform_and_write_content(original_content, output_file_path)


def count_true_and_false_positives_top100(predicted_lineages, known_lineages):
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


def plot_weekly_precision(precisions, file_path, title):
    """
    This function takes a list of precision values for each week and creates a line plot using Seaborn.

    :param precisions: List of precision values (float or int) for each week.
    """
    # Create a DataFrame with the precision values
    data = pd.DataFrame({'Week': range(len(precisions)), 'Precision': precisions})

    # Create a line plot
    sns.barplot(x='Week', y='Precision', data=data)

    # Add titles and labels
    plt.title('Weekly Precision')
    plt.xlabel('Week')
    plt.ylabel('Precision')

    # Save the plot
    plt.savefig(file_path + title)


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


def covered_area(measure_sensitivity):
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
        if k == 'unknown':
            continue
        indices_k = np.where(measure_sensitivity_np == k)[0]  # Find indices where my variant is
        total = np.array(list(map(int, measure_sensitivity_np[indices_k, 1])))
        predicted = np.array(list(map(int, measure_sensitivity_np[indices_k, 2])))
        week_analysis = np.array(list(map(int, measure_sensitivity_np[indices_k, 3])))
        first_prediction_index = np.where(predicted > 0)[0]  # Find index of first prediction
        if len(first_prediction_index) == 0:
            continue
        first_prediction_week = min(list(week_analysis[first_prediction_index]))
        index_first_prediction_week = np.where(
            week_analysis == first_prediction_week)  # Find the index of the week and sum the second column
        total_lineage = total[index_first_prediction_week]
        all_weeks_indices = np.where(Weeks == first_prediction_week)
        total_seq_week = np.sum(total_official[all_weeks_indices])
        area = total_lineage / total_seq_week
        final_area.append([k, area])
        only_area.append(area)

    return final_area, only_area


def write_precision(precision, week, n=100):
    # First, we filter the precision values based on the condition.
    filtered_precision_week = [precision[i] for i in range(len(week)) if week[i] > n]  # Paper

    # Median
    median_precision_week = np.mean(filtered_precision_week)
    q1_prec_week, q3_prec_week = np.percentile(filtered_precision_week, [25, 75])

    return [['The median is ' + str(median_precision_week), 'The 25th percentile is ' + str(q1_prec_week),
             'The 75th percentile is ' + str(q3_prec_week)]]
