import argparse
import logging
from sklearn.model_selection import ParameterGrid
import gc
from utils import *
import warnings
import tensorflow as tf
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    # Ignore all warnings
    warnings.filterwarnings('ignore')

    # Memory Control
    gc.enable()

    # GPU
    strategy = tf.distribute.MirroredStrategy()

    # Variables
    # Definition of a list for utilization throughout the code.
    prediction_lineages = []
    summary_lineages = []  # A list to store a summary for each lineage and for each week.
    number_of_feature = []  # Number of features.
    results_fine_tune = []
    fractions_100 = []  # A list to store predictions for the top 100 sequences with higher mean squared error (MSE).
    ind_prc = 0  # Counter
    summary_100_anomalies = []  # List containing the number of sequences considered as anomalies for each lineage and for each week of simulation.
    summary_100_anomalies_percentage = []  # List containing the percentage of sequences considered as anomalies for each lineage and for each week of simulation.
    precision_top_100 = []  # Store precision for top 100
    precision_overall = []  # Store precision overall
    lineages_in_week = []  # Store numer of FDLs during the simulation
    mean_positions = []  # Store mean position
    mrrs = []  # Mean Reciprocal Rank (MRR)
    precisions_top_25_percent = []  # precision_top_25
    precisions_top_50_percent = []  # precision_top_50
    precisions_top_75_percent = []  # precision_top_75
    df_lin_blosum = pd.DataFrame(columns=['Lineages_TP', 'Blosum_score', 'Sequence', 'TP_FP', 'Week'])
    ## Path to read the dataset
    dir_week = args.dir_week
    metadata_path = args.metadata_path
    metadata_2_path = args.metadata_2_path
    new_class_correction_path = args.new_class_correction_path
    fasta_path = args.fasta_path
    treshold_blosum = args.treshold_blosum

    metadata = pd.read_csv(metadata_path) # Read metadata filtered file generated by Data_Filtration_kmers.py
    metadata_2 = pd.read_csv(metadata_2_path) # Read metadata filtered file generated by Data_Filtration_kmers.py
    new_class_correction = pd.read_csv(new_class_correction_path)

    # Load all the Sequences
    dict_seq = read_protein_sequences_header(fasta_path)
    seq_origin = origin_spike(dict_seq)

    ## Columns in metadata
    col_class_lineage = 'Pango.lineage'
    col_submission_date = 'Collection.date'
    col_lineage_id = 'Accession.ID'

    ## Processing of Data
    valid_lineage, valid_lineage_prc, dictionary_lineages_week, lineages_know = lineages_of_interest() # Function that return the Future Dominant Lineages (FDLs) of Interest.

    ## Retraining Week
    retraining_week, retraining_week_false_positive = retraining_weeks() # Return retraining weeks.

    ## K-mers
    header = pd.read_csv(dir_week + '/1/EPI_ISL_529217.csv', nrows=1)
    features = header.columns[1:].tolist()  # k-mers
    print('-----------------------------------------------------------------------------------')
    print('The total number of k-mers is : ' + str(len(features)))
    print('-----------------------------------------------------------------------------------')

    path_save_file = args.path_save_file

    ## Lineage of interest
    lineage_of_interest_train = valid_lineage # Return the FDLs present in the dataset.

    ## Logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(path_save_file + '/Autoencode_performance.log', 'w+'),
                            logging.StreamHandler()
                        ])

    ## Training week
    starting_week = 1 # First week of training.

    ## Loading first training set
    df_trainstep_1, train_w_list = load_data(dir_week, [starting_week]) # First training set.
    train_step1 = df_trainstep_1.iloc[:, 1:len(df_trainstep_1.columns)].to_numpy()

    ## Filter the features of models
    sum_train = np.sum(train_step1, axis=0)
    keepFeature = sum_train / len(train_step1)
    i_no_zero = np.where(keepFeature >= 0.01)[0] # Retain the features that differ from 0 by at least N%. This approach ensures that only the most representative features are kept.
    features_no_zero = [features[i] for i in i_no_zero]  # features of model
    write_feature(features_no_zero, path_save_file, '/Features.txt')

    print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('The features of the model are :' + str((len(i_no_zero))))
    print('---------------------------------------------------------------------------------------------------------------------------------------------------------')

    ## Training set
    y_train_initial = metadata[metadata[col_lineage_id].isin(df_trainstep_1.iloc[:, 0].tolist())][col_class_lineage] # Elements of training set.
    y_train_class = map_lineage_to_finalclass(y_train_initial.tolist(), lineage_of_interest_train)  # Class of training set.
    counter_i = Counter(y_train_initial)

    ## Filtering out features with all zero
    train_step_complete_rw = train_step1
    train = train_step1[:, i_no_zero] # Select the representative features.
    lineages_train = np.array(y_train_initial.tolist()) # Type of lineages.

    tf.random.set_seed(10)
    ## Creation of  Autoencoder models

    # Parameters
    nb_epoch = args.nb_epoch
    batch_size = args.batch_size
    input_dim = train.shape[1]  # num of columns
    encoding_dim = 1024
    hidden_dim_1 = int(encoding_dim / 2)  # 512
    hidden_dim_2 = int(hidden_dim_1 / 2)  # 256
    hidden_dim_3 = int(hidden_dim_2 / 2)  # 128
    hidden_dim_4 = int(hidden_dim_3 / 2)  # 64
    hidden_dim_5 = int(hidden_dim_4 / 2)  # 32
    reduction_factor = 1e-7

    p_grid = {'nb_epoch': [nb_epoch], 'batch_size': [batch_size], 'input_dim': [input_dim], 'encoding_dim': [encoding_dim],
              'hidden_dim_1': [int(encoding_dim / 2)], 'hidden_dim_2': [hidden_dim_2], 'hidden_dim_3': [hidden_dim_3],
              'hidden_dim_4': [hidden_dim_4], 'hidden_dim_5': [hidden_dim_5], 'Reduction_factor': [reduction_factor]}
    all_combo = list(ParameterGrid(p_grid))

    with strategy.scope(): # Using GPU
        autoencoder = model(input_dim, encoding_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, hidden_dim_4, hidden_dim_5, reduction_factor, path_save_file) # Project the Deep Learning Model.
    for combo in all_combo[0:1]:
        logging.info("---> Autoencoder - Param: " + str(combo))
        y_test_dict_variant_type = {}
        y_test_dict_finalclass = {}
        y_test_dict_predictedclass = {}
        history,cost = autoencoder_training_GPU(autoencoder, train, train, nb_epoch, batch_size) # Training the model. (cost is constatnt for the treshold)
        autoencoder.save(path_save_file + '/Autoencoder_models.h5') # saving model in h5 format.
        print('The model is trained and saved !')
        info, mse_tr = test_normality(autoencoder, train) # Compute the MSE in the training set. This is important to define the threshold for the anomaly detection.
        ## SIMULATION
        for week in range(1, 198): # Simulation weeks
            if week in retraining_week:
                logging.info('----> RETRAINING <-----')
                ind_prc += 1
                # Creation of new training set to train the model.
                # Filter out unrepresentative features
                train_model_value = train_step_complete_rw
                classes = lineages_train
                sum_train = np.sum(train_model_value, axis=0)
                keepFeature = sum_train / len(train_model_value)
                i_no_zero = np.where(keepFeature > 0.01)[0]
                features_no_zero = [features[i] for i in i_no_zero]  # features of model
                write_feature(features_no_zero, path_save_file, '/Features.txt')
                number_feature = len(i_no_zero)

                print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('features of model :' + str((len(i_no_zero))))
                print('---------------------------------------------------------------------------------------------------------------------------------------------------------')

                train_model_value = train_model_value[:, i_no_zero] # Filter training set with the features of model.
                index_raw = find_indices_lineage_per_week_sublineage(classes, week, dictionary_lineages_week)  # return index to create the new training set
                train_model_value = train_model_value[index_raw, :]
                np.random.shuffle(train_model_value)
                number_of_feature.append(number_feature) # Store the number of features in the retraining week.

                batch_size = 512
                input_dim = train_model_value.shape[1]
                with strategy.scope():
                    autoencoder = model(input_dim, encoding_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, hidden_dim_4, hidden_dim_5,
                          reduction_factor, path_save_file) # Creation of the AutoEncoder model.
                history,cost = autoencoder_training_GPU(autoencoder, train_model_value, train_model_value, nb_epoch, batch_size) # retraining the model.
                autoencoder.save(path_save_file + '/Autoencoder_models.h5')
                print('the model is trained and saved')

                # Compute MSE in the training set
                info, mse_tr = test_normality(autoencoder, train_model_value)

                # Compute Blosum score in the True positive known
                treshold_blosum = count_true_and_false_positives_with_blosum_score(df_lin_blosum, lineages_know[ind_prc])
                train_model_value = []
                classes = []
            logging.info("# Week " + str(starting_week + week))
            print("# Week " + str(starting_week + week))

            ## Loading test set
            df_teststep_i, test_w_list = load_data(dir_week, [starting_week + week]) # Test set.
            test_step_i = df_teststep_i.iloc[:, 1:len(df_teststep_i.columns)].to_numpy() # transform in numpy.
            id_identifier = df_teststep_i.iloc[:, 0].to_list() # Sequence Identifier.
            test_step_complete_rw = test_step_i # (rw = retraining week)
            test_step_i = test_step_i[:, i_no_zero] # feature selections
            y_test_step_i = get_lineage_class(metadata, id_identifier) # type of lineages present in test set.
            print(y_test_step_i)
            # last class
            y_test_step_i = update_class_list(new_class_correction, id_identifier, y_test_step_i)
            print(y_test_step_i)
            number_of_true_positives = true_lineages_week(y_test_step_i, valid_lineage_prc[ind_prc])
            lineages_in_week.append(number_of_true_positives)
            lineages_test = y_test_step_i # lineages in the week [array]
            y_test_dict_variant_type[starting_week + week] = y_test_step_i
            y_test_fclass_i = map_lineage_to_finalclass(y_test_step_i, valid_lineage_prc[ind_prc])  # return the class of sequences present in the test set. (-1->FDLs, 1->No FDLs).
            i_voc = np.where(np.array(y_test_fclass_i) == -1)[0]
            y_test_dict_finalclass[starting_week + week] = y_test_fclass_i
            lineage_dict = Counter(y_test_step_i) # Dictionary that contains the lineages present in the test set.

            ## Model Prediction
            test_x_predictions = autoencoder.predict(test_step_i) # predictions

            ## Threshold
            mse = np.mean(np.power(test_step_i - test_x_predictions, 2), axis=1) # Mean Square Error (MSE).
            error_df = pd.DataFrame({'Reconstruction_error': mse})
            threshold_fixed = np.mean(mse_tr) + float(cost) * np.std(mse_tr) # Threshold for anomaly detection.
            print('Threshold is : ' + str(threshold_fixed))
            y_test_i_predict = [-1 if e >= threshold_fixed else 1 for e in error_df.Reconstruction_error.values]
            y_test_i_predict = np.array(y_test_i_predict)

            ## Filter
            y_test_i_predict, mse = lookup(y_test_i_predict, y_test_step_i, lineages_know[ind_prc], mse)

            ## The k-mers importance
            i_anomaly = np.where(y_test_i_predict == -1)[0]
            id_anomaly_seq = [id_identifier[i] for i in i_anomaly] # Gisaid ID for the sequences
            features_no_zero = [features[i] for i in i_no_zero] # features of model
            selection_kmers(test_x_predictions, test_step_i, features_no_zero, y_test_i_predict, id_identifier, path_save_file + '/Summary_' + str(starting_week + week) + '.csv') # this function identifies kmers that have not been reproduced correctly by the model.

            ## Understand the error and save the outputs
            lineages_error_test = np.array(y_test_step_i)

            # Selection of sequences considered as anomalies by the model.
            mse_top100_anomaly = mse[i_anomaly]
            lineage_top100_anomaly = lineages_error_test[i_anomaly]

            # Compute precision overall
            true_positive_overall, false_positive_overall = count_true_and_false_positives_overall(lineage_top100_anomaly, valid_lineage_prc[ind_prc])
            precision_overall.append(true_positive_overall / (true_positive_overall + false_positive_overall + 0.001))
            plot_weekly_precision(precision_overall, path_save_file, '/precision_overall.png')

            # Select the top 100 and sort mse
            size = 100
            if len(i_anomaly) < 100:
                size = len(i_anomaly)
            top_indices_100 = mse_top100_anomaly.argsort()[-size:][::-1] # sort the MSE
            lineages_predicted_top_100 = lineage_top100_anomaly[top_indices_100] # Find the lineages
            mse_top100_anomaly_of = [mse_top100_anomaly[i] for i in top_indices_100] # mse of top100
            Id_lineages_anomaly = [id_anomaly_seq[i] for i in top_indices_100] # ID_TOP100

            ## Filtering with the biological knowledge
            prediction = list(-np.ones(size))
            prediction_filtering = lookup_post(prediction, lineages_predicted_top_100, lineages_know[ind_prc]) # Filter the prediction

            # Filtering with biological data (BLOSUM)
            seq_anomaly_week = [dict_seq[Id_lineages_anomaly[i]] for i in range(len(Id_lineages_anomaly))]

            # Compute BLOSUM62 score
            blosum62_score = calculate_blosum_scores(seq_origin, seq_anomaly_week)
            prediction_filtering = adjust_predictions_based_on_blosum(blosum62_score, prediction_filtering, treshold_blosum)

            ## Find the anomalies after filtering
            prediction_filtering = np.array(prediction_filtering)
            index_anomaly_filter = np.where(prediction_filtering == -1)[0] # Find the anomalies.
            seq_anomaly_week_filter = [seq_anomaly_week[i] for i in index_anomaly_filter]
            lineages_predicted_top_100 = lineages_predicted_top_100[index_anomaly_filter]
            mse_top100_anomaly_of_filt = [mse_top100_anomaly_of[i] for i in index_anomaly_filter]

            ## Compute metrix position
            result_position = analyze_lineages(lineages_predicted_top_100, mse_top100_anomaly_of_filt, valid_lineage_prc[ind_prc])
            mean_positions.append(result_position["mean_position"])
            mrrs.append(result_position["mrr"])
            precisions_top_25_percent.append(result_position["precision_top_25"])
            precisions_top_50_percent.append(result_position["precision_top_50"])
            precisions_top_75_percent.append(result_position["precision_top_75"])

            plot_evaluation_metrics(mean_positions, mrrs, precisions_top_25_percent, precisions_top_50_percent, precisions_top_75_percent, path_save_file + '/measure_position')

            lineages_counter_top_100 = Counter(lineages_predicted_top_100) # Count the anomalies identified by the model
            total_100 = sum(lineages_counter_top_100.values())
            lineage_percentage_100 = {k: (v / total_100) * 100 for k, v in lineages_counter_top_100.items()}

            # list prediction during the simulation week
            summary_100_anomalies.append([week, lineages_counter_top_100])
            summary_100_anomalies_percentage.append([week, lineage_percentage_100])

            # Write the file in txt the prediction
            with open(path_save_file + '/TOP_100_FILTERING.txt', 'w') as file:
                for elemento in summary_100_anomalies:
                    file.write(str(elemento) + '\n')

            # Write the file in txt the prediction precision
            with open(path_save_file + '/TOP_100_FILTERING_PERCENTAGE.txt', 'w') as file:
                for elemento in summary_100_anomalies_percentage:
                    file.write(str(elemento) + '\n')

            # Compute the precision top 100
            true_positive_top100, false_positive_top100, vector_tp_fp = count_true_and_false_positives_top100(lineages_predicted_top_100, valid_lineage_prc[ind_prc])
            precision_top_100.append(true_positive_top100 / (true_positive_top100 + false_positive_top100 + 0.001))
            plot_weekly_precision(precision_top_100, path_save_file, '/precision_top100')

            # Memorize the blosum score associated at True Positive
            index_tp = np.where(vector_tp_fp == 1)[0]
            lin_TP = [lineages_predicted_top_100[i] for i in index_tp]
            Blosum_TP = [blosum62_score[i] for i in index_tp]

            if len(lineages_predicted_top_100) > 0:
                for i in range(len(lineages_predicted_top_100)):
                    new_raw = {'Lineages_TP': lineages_predicted_top_100[i], 'Blosum_score': blosum62_score[i], 'Sequence': seq_anomaly_week_filter[i], 'TP_FP': vector_tp_fp[i], 'Week': starting_week + week}
                    df_lin_blosum = df_lin_blosum.append(new_raw, ignore_index=True)
                    df_lin_blosum.to_csv(path_save_file + '/Results.csv', index=False)

            # Training set for retraining
            train_step_complete_rw = np.concatenate((train_step_complete_rw, test_step_complete_rw)) # (rw = retraining week)
            lineages_train = np.concatenate((lineages_train, lineages_test)) # list that contains the lineages present in training set
            y_test_dict_predictedclass[starting_week + week] = y_test_i_predict
            y_test_voc_predict = np.array(y_test_i_predict)[i_voc]

            print('Analysys is done')

            for k in lineage_dict.keys():
                i_k = np.where(np.array(y_test_step_i) == k)[0]
                logging.info('Number of ' + k + ' lineage:' + str(len(i_k)) + '; predicted anomaly=' + str(
                    len([x for x in y_test_i_predict[i_k] if x == -1])))
                print('Number of ' + k + ' lineage:' + str(len(i_k)) + '; predicted anomaly=' + str(
                    len([x for x in y_test_i_predict[i_k] if x == -1])))
                # Store the file.
                h = len([x for x in y_test_i_predict[i_k] if x == -1])
                partial_summary = [k, h, week] # The list contains : [Name of lineage, Number of lineage sequences predicted as anomalies, week of simulation].
                prediction_lineages.append(partial_summary) # Store the partial summary
                complete_summary_lineages = [k, len(i_k), h, week]  # The list contains : [Name of lineage, Total number of lineage sequences in the week of simulation, Number of lineage sequences predicted as anomalies, week of simulation].
                summary_lineages.append(complete_summary_lineages) # Store complete summary.

        # saving results for this comb of param of the oneclass_svm
        results = {'y_test_variant_type': y_test_dict_variant_type,
                   'y_test_final_class': y_test_dict_finalclass,
                   'y_test_predicted_class': y_test_dict_predictedclass}
        results_fine_tune.append(results)


    ## Week before
    distance = weeks_before(summary_lineages, valid_lineage)
    distance_list = list(distance)
    with open(path_save_file + '/distance_prediction.txt', 'w') as file:
        for elemento in distance_list:
            file.write(str(elemento) + '\n')
    print('-----------------------------------------Prediction weeks before threshold-------------------------------------------------------')
    print(distance)
    print('---------------------------------------------------------------------------------------------------------------------------')

    ## Percentage Area
    area_discover, only_area = covered_area(summary_lineages, valid_lineage)
    area_median = np.median(only_area)
    q1, q3 = np.percentile(only_area, [25, 75])
    area_tot = [q1, area_median, q3]
    measure = ['25th percentile', '50th percentile', '75th percentile']
    with open(path_save_file + '/median_area.txt', 'w') as file:
        for i in range(len(area_tot)):
            file.write(measure[i] + ': ' + str(area_tot[i]) + '\n')

    print('-----------------------------------------Area Discovery-------------------------------------------------------')
    print(area_discover)
    print('---------------------------------------------------------------------------------------------------------------------------')

    ## Write precision
    summary_precision_top100 = write_precision(precision_top_100, lineages_in_week)
    with open(path_save_file + '/precision_top100.txt', 'w') as file:
        for i in range(len(summary_precision_top100)):
            file.write(str(summary_precision_top100[i]) + '\n')

    # Write MRRS
    for i in range(len(mrrs)):
        if mrrs[i] is None:
            mrrs[i] = 0

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(mrrs, marker='o', linestyle='-', color='blue', label='Initial Data')
    plt.title('MRR')
    plt.xlabel('MRR')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

    mrrs_np = np.array(mrrs)
    mrrs_np_mean = np.mean(mrrs_np[np.where(np.array(lineages_in_week) > 0)])

    print('MRRS_MEAN : ')
    print(mrrs_np_mean)

    with open(path_save_file + '/mrrs.txt', 'w') as file:
        file.write('MRRS_MEAN : \n')
        file.write(str(mrrs_np_mean) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepAutoCov Simulation Script")
    parser.add_argument('--dir_week', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--metadata_path', type=str, required=True, help="Path to the metadata file")
    parser.add_argument('--metadata_2_path', type=str, required=True, help="Path to the second metadata file")
    parser.add_argument('--new_class_correction_path', type=str, required=True, help="Path to the new class correction file")
    parser.add_argument('--fasta_path', type=str, required=True, help="Path to the fasta file")
    parser.add_argument('--treshold_blosum', type=int, default=6722, help="Threshold for BLOSUM score")
    parser.add_argument('--path_save_file', type=str, required=True, help="Path to save the outputs")
    parser.add_argument('--nb_epoch', type=int, default=50, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training")
    args = parser.parse_args()

    main(args)
