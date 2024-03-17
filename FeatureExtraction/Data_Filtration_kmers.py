# Import the Library and function
from optparse import OptionParser
from utils import * # utils.py contains the functions used in this code
import statistics as st
# example of continent = ['Denmark', 'France', 'United Kingdom', 'USA', '/', 'Denmark']

def main(options):
    continent=options.continent_list
    for l in continent:
        print("\033[1m Read File Metadata and Fasta \033[0m")
        sequences,headers = read_protein_sequences_header(str(options.fasta_path)) # Read the FASTA file (sequences) that contains the aminoacid sequences.
        metadata = pd.read_csv(str(options.csv_path))
        print(len(sequences))

        # Merging dataset
        # Read the CSV file (metadata) that contains information about the sequences from the GISAID database.
        metadata.iloc[:, 4] = metadata.iloc[:, 4].astype(type(headers[0]))
        metadata_ordinato = pd.DataFrame(headers, columns=['ID']).merge(metadata, left_on='ID', right_on=metadata.columns[4])
        metadata_ordinato['Sequences'] = sequences
        metadata = metadata_ordinato.drop('ID', axis=1)
        sequences = metadata["Sequences"].values.tolist()
        metadata = metadata.drop(['Sequences'], axis=1)
        #metadata.to_csv(str(options.save_path)+"/dataset_official.csv", index=False)
        metadata = metadata.to_numpy()

        print(len(metadata))
        print("\033[1m Metadata file and Fasta have been uploaded \033[0m")

        print("\033[1m Remove the asterisks at the end of the sequence \033[0m")
        sequences = [remove_asterisks(s) for s in sequences] # Remove the asterisks at the end of the sequence.
        print("\033[1m Asterisks have been removed \033[0m")

        print("\033[1m Filter lineages by country of provenance \033[0m")
        metadata_nation, index_nation = select_rows_dataset(metadata, l) # Select the rows of the dataset (metadata) that belong to the chosen country (specified as a command line parameter).
        sequences_nation = [sequences[i] for i in index_nation] # Select the sequences (metadata) that belong to the chosen country (specified as a command line parameter).
        print("\033[1m Lineages were filtered by country of provenance \033[0m")

        Dimension = len(sequences_nation) # Calculate the number of sequences present in a nation.
        print('\033[1m The number of spike proteins : ' + str(Dimension) + '\033[0m')
        # Length delle sequenze
        Length = []
        for sequence in sequences_nation:
            Length.append(len(sequence)) # This vector stores the length of each sequence.

        print(Length)
        print('\033[1m Filter sequences with length less than ' + str(options.min_length) +'\033[0m')
        sequences_filtering_min_limit = [x for i, x in enumerate(sequences_nation) if Length[i] >= int(options.min_length)] # Select sequences that are longer than a specified threshold (an integer value set via a command-line parameter). Sequences shorter than this threshold are considered incorrect.
        index = [i for i, x in enumerate(Length) if x >= int(options.min_length)] # Compute the indices of valid sequences.
        print('\033[1m Update the Metadata file \033[0m')
        metadata_filter_min_limit = metadata_nation[index] # Update the dataset.
        Dimensione_fil_min_limit = len(sequences_filtering_min_limit) # Compute the number of valid sequences in the dataset.
        print('The number of spike proteins after deleting sequences with length less than '+ str(options.min_length)+' is: ' + str(
            Dimensione_fil_min_limit))

        print('\033[1m Compute the length of valid sequences ' + str(options.min_length)+'\033[0m')
        Length_filtering_min_limit = []
        for sequence in sequences_filtering_min_limit:
            Length_filtering_min_limit.append(len(sequence))


        print('\033[1m Evaluation of valid sequence contained in the database\033[0m')
        valid_sequences, invalid_sequences, valid_indices, invalid_indices = validate_sequences(
            sequences_filtering_min_limit) # Identify sequences that contain valid characters (amino acids).
        print('\033[1m Results : \033[0m')
        print("Ther'are " + str(len(valid_sequences)) + ' valid sequence in the database')
        print("Ther'are " + str(len(invalid_sequences)) + ' not valid sequence in the database')

        # Aggiorno il file metadata
        print('\033[1m Update the metadata  \033[0m')
        metadata_valid_indices = metadata_filter_min_limit[valid_indices] # Update the metadata file with the correct sequences.
        metadata_valid_invalid_indices = metadata_filter_min_limit[invalid_indices]

        Length_filtering_min_1000_valid = []
        for sequence in valid_sequences:
            Length_filtering_min_1000_valid.append(len(sequence)) # Compute the length of valid sequences

        if not Length_filtering_min_1000_valid:
            print('There are not the valid sequences in the database. The analysis is stopped')
            break

        print('\033[1m filter the sequences by the length included in the median \033[0m')
        extreme_inf = st.median(Length_filtering_min_1000_valid) - int(options.med_limit) # Minimum acceptable length for protein spikes.
        extreme_sup = st.median(Length_filtering_min_1000_valid) + int(options.med_limit) # Maximum acceptable length for protein spikes.
        index, valid_sequence = filter_sequences(valid_sequences, extreme_inf, extreme_sup) # Filter the sequences with the length within the defined range.
        metadata_valid_indices_length = metadata_valid_indices[index] # Uptade the metadata with the sequences with the length within the defined range.
        print(str(len(valid_sequence)) +' sequences with length between ' + str(
            extreme_inf) + ' and ' + str(extreme_sup))

        print("\033[1m Filter sequences by dates \033[0m")
        metadata_off, sequences_off, metadata_not_off, sequences_not_off = filter_row_by_column_length_sostitution(
            metadata_valid_indices_length, valid_sequence, 5, 10) # Filter sequences by keeping the sequences that have the correct submission dates.
        print("\033[1m The number of sequences filtered with dates is :\033[0m" + str(len(metadata_off)))

        print("\033[1m Reordering the metadata file \033[0m")
        metadata_tot = insert_sequence_as_column(metadata_off, metadata_off[:, 5], sequences_off) # Reorder the dataset according to submission dates.

        sequences = list(metadata_tot[:, 24])  # sequences
        metadata = metadata_tot[:, 0:23]  # metadata


        print("\033[The number of simulation week : \033[0m")
        indices_by_week = split_weeks(metadata[:, 5])
        print(len(indices_by_week)) # Number of simulation week
        seq_week = []
        for i in range(0, len(indices_by_week)):
            seq_week.append(len(indices_by_week[i]))
        print(seq_week)

        if l=='/':
            l='Global'
        write_csv_dataset(metadata, l, str(options.save_path)) # Create a CSV file for new dataset.


        print('\033[1m Calculation of k-mers\033[0m')
        k = 3 # k for the k-mers
        kmers = calculate_kmers(valid_sequence, k) # Calculate the possible k-mers of sequences.
        kmers_unique = list(set(kmers))

        import os
        for i in range(0, len(indices_by_week)):
            indices = indices_by_week[i] # Index of sequences present in the week of simulation.
            sequences_for_week = []
            identifier_for_week = []
            week = i + 1
            os.makedirs(str(options.save_path) + '/' + str(week)) # Set the path where saving the files.
            for m, index in enumerate(indices):
                sequences_for_week.append(sequences[index]) # Sequences present in the week of simulation.
                identifier_for_week.append(metadata[index, 4]) # Identifiers present in the week of simulation.
            for h, seq in enumerate(sequences_for_week):
                format_csv(seq, identifier_for_week[h], kmers_unique, k, week, l, str(options.save_path)) # Write the csv file for the dataset.

        # Creating the dataset for the simulation
        import os
        import csv

        csv_directory = str(options.save_path) # Fix the directory of csv

        # Loop through all subdirectories and files in the main folder.
        for root, directories, files in os.walk(csv_directory):
            for directory in directories:
                # Create a text file and open it in append mode.
                txt_file = os.path.join(root, directory, "week_dataset.txt")
                with open(txt_file, "a") as output_file:
                    # Iterate over each file in the current directory.
                    for filename in os.listdir(os.path.join(root, directory)):
                        # Check if the file is a CSV file.
                        if filename.endswith(".csv"):
                            # Construct the full path to the CSV file.
                            csv_file = os.path.join(root, directory, filename)
                            # Open the CSV file using the csv library and read each line.
                            with open(csv_file, "r") as input_file:
                                reader = csv.reader(input_file)
                                next(reader)  # Skip the first line (usually headers).
                                for row in reader:
                                    # Write each row to the text file.
                                    output_file.write(",".join(row) + "\n")

    print('\033[1m The dataset was created in the directory: ' + str(options.save_path) +'! \033[0m')
if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-f", "--fasta", dest="fasta_path",

                      help="path of FASTA file", default="spikes.fasta")
    parser.add_option("-c", "--csv", dest="csv_path",

                      help="path of CSV file", default="metadata.csv")

    parser.add_option("-n","--continent",dest="continent_list",
                      help="list of continents of interest",default=['/'])

    parser.add_option("-m", "--minlen ", dest="min_length",
                      help="minimum length of sequences", default=1000)

    parser.add_option("-l", "--median_limit ", dest="med_limit",
                      help="median range", default=30)

    parser.add_option("-p", "--path_save_file ", dest="save_path",
                      help="path where saving the output", default='')


    (options, args) = parser.parse_args()
    main(options)

