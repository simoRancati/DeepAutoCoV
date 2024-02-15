import pandas as pd
import csv
from datetime import datetime
import random
import numpy as np


def read_protein_sequences_header(file):
    """
    This function reads a FASTA file, extracts protein sequences, and their respective headers.

    Parameters:
    - file: The path to the FASTA file.

    Returns:
    - sequences: A list containing only the protein sequences found in the FASTA file.
    - headers: A list containing the headers for each protein sequence.
    """

    sequences = []  # Initialize an empty list to store protein sequences.
    headers = []    # Initialize an empty list to store headers of the sequences.

    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        current_sequence = ''  # A variable to hold the current sequence.

        for line in f:
            line = line.strip()  # Strip whitespace from the line.

            if line.startswith('>'):  # Check for header line.
                if current_sequence:  # If there's a current sequence, store it before starting a new one.
                    sequences.append(current_sequence)
                    current_sequence = ''

                # Add the header line to the headers list, excluding the first character ('>')
                headers.append(line[1:])

            else:
                current_sequence += line  # Append non-header lines to the current sequence.

        if current_sequence:  # After the last line, add the final sequence.
            sequences.append(current_sequence)

    return sequences, headers

def read_fasta(file):
    """
    This function reads a FASTA file and extracts the sequences.

    Parameters:
    - file: The path to the FASTA file.

    Returns:
    - sequences: A list containing all the sequences found in the FASTA file.
    """

    # Initialize an empty list to store the sequences.
    sequences = []

    # Open the file for reading.
    with open(file, 'r') as f:
        # Initialize a variable to hold the current sequence.
        current_sequence = ''
        # A flag to indicate if a sequence has started.
        started = False

        # Iterate over each line in the file.
        for line in f:
            # Strip the line of leading and trailing whitespace.
            line = line.strip()

            # Check if the line is a header line (starts with '>').
            if line.startswith('>'):
                # If a sequence has already started, add the current sequence to the list.
                if started:
                    sequences.append(current_sequence)
                # Set the flag to indicate that a sequence has started.
                started = True
                # Reset the current sequence.
                current_sequence = ''
            else:
                # If it's not a header line, add the line to the current sequence.
                current_sequence += line

        # After the last line, add the final sequence to the list if it's not empty.
        if current_sequence:
            sequences.append(current_sequence)

    # Return the list of sequences.
    return sequences

def read_csv(file):
    """
    This function reads a CSV (Comma-Separated Values) file and returns its contents.

    Parameters:
    - file: The path to the CSV file.

    Returns:
    - A NumPy array containing the data from the CSV file.
    """
    # Reads the CSV file using pandas and converts it to a NumPy array.
    return pd.read_csv(file).values

def read_tsv(file):
    """
    This function reads a TSV (Tab-Separated Values) file and returns its contents.

    Parameters:
    - file: The path to the TSV file.

    Returns:
    - A NumPy array containing the data from the TSV file.
    """
    # Reads the TSV file using pandas with a specified separator (tab) and converts it to a NumPy array.
    return pd.read_csv(file, sep='\t').values


def validate_sequences(sequences):
    """
    This function validates a list of sequences based on standard amino acid characters.

    Parameters:
    - sequences: A list of sequences (strings) to be validated.

    Returns:
    - valid_sequences: A list of valid sequences.
    - invalid_sequences: A list of invalid sequences.
    - valid_indices: A list of indices corresponding to valid sequences.
    - invalid_indices: A list of indices corresponding to invalid sequences.
    """

    # Initializing lists to store valid and invalid sequences, and their respective indices.
    valid_sequences = []
    invalid_sequences = []
    valid_indices = []
    invalid_indices = []

    # Iterating over each sequence and its index in the provided list of sequences.
    for index, seq in enumerate(sequences):
        # Initially assuming the sequence is valid.
        is_valid = True

        # Checking each amino acid in the sequence.
        for amino_acid in seq:
            # If the amino acid is not one of the standard amino acids, mark the sequence as invalid.
            if amino_acid not in "ACDEFGHIKLMNPQRSTVWY":
                is_valid = False
                break  # Exiting the loop as soon as an invalid amino acid is found.

        # If the sequence is valid, add it and its index to the valid lists.
        if is_valid:
            valid_sequences.append(seq)
            valid_indices.append(index)
        else:
            # If the sequence is invalid, add it and its index to the invalid lists.
            invalid_sequences.append(seq)
            invalid_indices.append(index)

    # Returning the lists of valid and invalid sequences, along with their indices.
    return valid_sequences, invalid_sequences, valid_indices, invalid_indices

def remove_asterisks(sequence):
    # This function is designed to remove asterisks (*) from the end of a given sequence.
    # Parameter:
    # sequence: A string representing the sequence from which asterisks should be removed.

    # The function uses Python's rstrip method on the sequence.
    # rstrip("*") removes all trailing asterisks (*) from the end of the sequence.
    # If there are no asterisks at the end, the sequence remains unchanged.
    return sequence.rstrip("*")

def filter_sequences(sequences, length_minimum, length_maximum):
    """
    This function filters sequences based on their length.
    It returns sequences that fall within a specified minimum and maximum length.

    Parameters:
    - sequences: A list of sequences to be filtered.
    - length_minimum: An integer representing the minimum acceptable length of a sequence.
    - length_maximum: An integer representing the maximum acceptable length of a sequence.

    Returns:
    - index: A list of indices of sequences that meet the length criteria.
    - sequences_valid: A list of sequences that are within the specified length range.
    """

    # Creates a list of indices for those sequences whose lengths are within the specified range.
    # This is done by enumerating over sequences and checking if the length of each sequence
    # falls between length_minimum and length_maximum.
    index = [i for i, seq in enumerate(sequences) if length_minimum <= len(seq) <= length_maximum]

    # Creates a list of sequences that are within the specified length range.
    # This is similar to the previous step, but instead of indices, the actual sequences are collected.
    sequences_valid = [seq for i, seq in enumerate(sequences) if length_minimum <= len(seq) <= length_maximum]

    # Returns two lists: one with the indices of the valid sequences and one with the valid sequences themselves.
    return index, sequences_valid


def calculate_kmers(sequences, k):
    """
    This function calculates k-mers from a list of sequences. A k-mer is a substring of length 'k' derived from a longer sequence.

    Parameters:
    - sequences: A list of sequences (strings) from which to derive the k-mers.
    - k: The length of each k-mer.

    Returns:
    - kmers: A list of all k-mers extracted from the input sequences.
    """

    # Initialize an empty list to store all k-mers.
    kmers = []

    # Iterate over each sequence in the provided list.
    for sequence in sequences:
        # Loop through the sequence to extract all possible k-mers.
        # The range is set up to stop at a point where a full-length k-mer can be obtained.
        for i in range(len(sequence) - k + 1):
            # Extract the k-mer starting at the current position 'i' and spanning 'k' characters.
            kmer = sequence[i:i + k]
            # Append the extracted k-mer to the list of kmers.
            kmers.append(kmer)

    # Return the list of all k-mers extracted from the input sequences.
    return kmers

def format_csv(seq, identifier, kmers_tot, k, week, l, path):
    """
    This function formats data into a CSV file, specifically for representing k-mers and their presence in a sequence.

    Parameters:
    - seq: The sequence from which k-mers are generated.
    - identifier: A unique identifier for the sequence.
    - kmers_tot: A list of all possible k-mers.
    - k: The length of each k-mer.
    - week: The week number, used in naming the output file.
    - l: An additional parameter, not currently used in the function (potentially for future use).
    - path: The path to the directory where the CSV file will be saved.

    Returns:
    - A string indicating that the operation is completed.
    """

    # Initialize a list for k-mers from the sequence and a binary list indicating their presence.
    kmers = []
    binary = [identifier]  # Start the binary list with the sequence identifier.

    kmers.append(None)  # Append a placeholder at the start of the k-mers list.

    # Generate k-mers from the input sequence and append them to the kmers list.
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        kmers.append(kmer)

    # Create a binary representation where 1 indicates the presence of a k-mer from kmers_tot in kmers.
    for km in kmers_tot:
        if km in kmers:
            binary.append(1)
        else:
            binary.append(0)

    # Add a placeholder at the beginning of kmers_tot.
    kmers_tot = [None] + kmers_tot

    # Write the k-mers and their binary representation to a CSV file.
    with open(str(path) + '/' + str(week) + '/' + str(identifier) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(kmers_tot)  # Write the total k-mers row.
        writer.writerow(binary)     # Write the binary representation row.

    return 'Done'  # Indicate that the function has finished.


def split_weeks(dates):
    """
    This function organizes a list of dates into weeks, grouping indices of dates by the week they belong to.

    Parameters:
    - dates: A list of dates as strings in the format 'YYYY-MM-DD'.

    Returns:
    - indices_by_week: A list of lists, where each sublist contains the indices of dates that fall into the same week.
    """

    # Convert string dates into datetime objects and sort them.
    date_objs = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    date_objs.sort()

    # Identify the start and end dates.
    start_date = date_objs[0]
    end_date = date_objs[-1]

    # Calculate the total number of weeks between the start and end dates.
    num_weeks = ((end_date - start_date).days // 7) + 1

    # Initialize a list of lists to store indices of dates by week.
    indices_by_week = [[] for _ in range(num_weeks)]

    # Iterate over each date and determine which week it belongs to.
    for i, date_obj in enumerate(date_objs):
        # Calculate the difference in days between the current date and the start date.
        days_diff = (date_obj - start_date).days

        # Determine the week number for the current date.
        week_num = days_diff // 7

        # Append the index of this date to the corresponding week's list.
        indices_by_week[week_num].append(i)

    # Return the list of indices grouped by week.
    return indices_by_week

def trimestral_indices(dates_list, m):
    """
    This function organizes dates into trimesters (a period of 'm' months) and returns the indices of dates in each trimester.

    Parameters:
    - dates_list: A list of dates as strings in the format 'YYYY-MM-DD'.
    - m: The number of months in each trimester.

    Returns:
    - A list of lists containing the indices of dates in each sorted trimester.
    """

    # Convert the string dates into datetime objects.
    dates = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in dates_list]

    # Create a dictionary to map each trimester (identified by year and trimester number) to a list of date indices in that trimester.
    trimestral_indices = {}
    for i, date in enumerate(dates):
        # Extract the year and calculate the trimester number based on the month.
        year = date.year
        trimester = (date.month - 1) // m + 1
        # Create a tuple key (year, trimester).
        key = (year, trimester)
        # If the key is not already in the dictionary, add it with an empty list.
        if key not in trimestral_indices:
            trimestral_indices[key] = []
        # Append the index of the date to the appropriate list in the dictionary.
        trimestral_indices[key].append(i)

    # Sort the keys (year, trimester) and create a list of lists of date indices, sorted by year and trimester.
    sorted_keys = sorted(trimestral_indices.keys())
    return [trimestral_indices[key] for key in sorted_keys]

def filter_row_by_column_length_sostitution(ndarray, string_list, col_index, target_len):
    """
    This function filters rows of a NumPy array and a parallel list based on the length of strings in a specified column
    of the array, after potentially modifying these strings to match a target length.

    Parameters:
    - ndarray: The NumPy array to be processed.
    - string_list: A list of strings parallel to the rows of the ndarray.
    - col_index: The index of the column in the ndarray to check and potentially modify the string length.
    - target_len: The desired target length for the strings after modification.

    Returns:
    - A tuple containing:
        1. The filtered ndarray with rows where the modified string length matches the target length.
        2. The filtered string_list corresponding to these rows.
        3. The inverse filtered ndarray with rows where the string length does not match the target.
        4. The inverse filtered string_list.
    """

    def check_and_add_suffix(s):
        """
        A nested function that checks the length of a string and, if necessary, appends a random suffix to it.
        """
        suffixes = ["-01", "-07", "-10", "-14", "-20", "-25", "-27"]
        if len(s) == 7:
            extended_s = s + random.choice(suffixes)
            if len(extended_s) == target_len:
                return extended_s
        return s

    # Apply the nested function to each string in the specified column of the ndarray.
    extended_strings = np.array([check_and_add_suffix(s) for s in ndarray[:, col_index]])

    # Create a boolean mask for rows where the modified string length matches the target length.
    mask = np.array([len(s) == target_len for s in extended_strings], dtype=bool)

    # Update the ndarray with the modified strings.
    ndarray[:, col_index] = extended_strings

    # Use the mask to filter the ndarray and string_list, and also create their inverse filtered versions.
    return ndarray[mask], [s for s, m in zip(string_list, mask) if m], ndarray[np.logical_not(mask)], [s for s, m in
                                                                                                       zip(string_list,
                                                                                                           mask) if
                                                                                                       not m]
def insert_sequence_as_column(data, dates, sequence):
    """
    It inserts the amino acid sequence as a column of an ndarray that contains the metadata,
    then sorting the ndarray by dates (in ascending order).

    Args:
        data (ndarray): the ndarray containing the metadata
        dates (list): the list of dates corresponding to each row in the ndarray.
        sequence (list): the list of the amino acid sequence to be entered as a column.

    Returns:
        ndarray: the ndarray sorted by dates, with the amino acid sequence inserted as a column.
    """
    # Transform date in object datetime
    date_objs = np.array([datetime.strptime(date, '%Y-%m-%d') for date in dates])

    # Add the sequence column as the last column of the ndarray
    data_with_sequence = np.column_stack((data, sequence))

    # Add a column with datetime objects of the dates
    data_with_dates = np.column_stack((data_with_sequence, date_objs))

    # Sort the ndarray by dates
    sorted_indices = np.argsort(data_with_dates[:, -1])
    sorted_data = data_with_dates[sorted_indices]

    # Delete the date column
    sorted_data = np.delete(sorted_data, -1, axis=1)

    return sorted_data

def write_csv_dataset(array, l, path_to_save):
    """
    This function writes a dataset to a CSV file.

    Parameters:
    - array: The dataset to be written, which is assumed to be a collection of rows (like a list of lists or a 2D array).
    - l: A label or identifier used in the naming of the output CSV file.
    - path_to_save: The file path where the CSV file will be saved.

    The function defines column names for the CSV file, opens a new CSV file in write mode at the specified path,
    and writes the data from the array to this file, using the defined column names as the header.
    """

    # Define the column names for the CSV file as a list of strings.
    name_columns = ['Virus.name', 'Not.Impo', 'format', 'Type', 'Accession.ID',
                    'Collection.date', 'Location', 'Additional.location.information',
                    'Sequence.length', 'Host', 'Patient.age', 'Gender', 'Clade',
                    'Pango.lineage', 'Pangolin.type', 'Variant', 'AA.Substitutions',
                    'Submission.date', 'Is.reference.', 'Is.complete.', 'Is.high.coverage.',
                    'Is.low.coverage.', 'N.Content']

    # Open a new CSV file in write mode at the specified path.
    with open(path_to_save + '/filtered_metadatataset_' + l + '.csv', "w", newline="") as csvfile:
        # Create a CSV writer object with comma as the delimiter.
        writer = csv.writer(csvfile, delimiter=",")

        # Write the first row (header) of the CSV file using the column names.
        writer.writerow(name_columns)

        # Iterate through each row in the input array.
        # For each row in the array, write it to the CSV file.
        for row in array:
            writer.writerow(row)

def select_rows_dataset(ndarray, country):
    """
    This function selects rows from a NumPy array (ndarray) based on whether they contain a specified country in a specific column.

    Parameters:
    - ndarray: The dataset in the form of a NumPy array from which rows are to be selected.
    - country: A string representing the country based on which rows are to be filtered.

    Returns:
    - A tuple containing:
        1. A NumPy array of selected rows that contain the specified country.
        2. A list of indices corresponding to these selected rows.
    """

    # Initialize empty lists to store the selected rows and their indices.
    selected_rows = []
    selected_indices = []

    # Iterate over each row in the dataset along with its index.
    for index, row in enumerate(ndarray):
        # Check if the 7th element of the row (index 6) is a string and contains the specified country.
        # The 7th element is assumed to hold country information in the dataset.
        if isinstance(row[6], str) and country in row[6]:
            # If the condition is met, append the entire row to 'selected_rows'.
            selected_rows.append(row)
            # Also, append the current index to 'selected_indices'.
            selected_indices.append(index)

    # Return the filtered rows as a NumPy array and the list of their indices.
    return np.array(selected_rows), selected_indices