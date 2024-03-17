# Load necessary libraries
library("Biostrings")
library("stringr")
library("vroom")

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
metadata_file_path <- args[1]
fasta_file_path <- args[2]
output_sequences_path <- args[3]
output_metadata_path <- args[4]

# Read the sequences
spikes = readAAStringSet(fasta_file_path)

# Process the names
original_spike_names = names(spikes)
names(spikes) = unlist(lapply(str_split(names(spikes), '\\|'), '[[', 4))

# Read metadata file
metadata <- vroom(metadata_file_path)
spec(metadata)

# Filter metadata based on specified conditions
metadata = metadata[metadata$`Is high coverage?` == TRUE &
                       !is.na(metadata$`Is high coverage?`) & 
                       metadata$`Is complete?` == TRUE &
                       metadata$Host == "Human" &
                       !is.na(metadata$`Is complete?`) &
                       !is.na(metadata$`Pango lineage`) &
                       !is.na(metadata$`Submission date`) &
                       str_length(metadata$`Submission date`) == 10,]

# Exclude incongruent instances
spikes = spikes[names(spikes) %in% metadata$`Accession ID`] 
metadata = metadata[metadata$`Accession ID` %in% names(spikes),] 

# Save the filtered sequences and metadata
writeXStringSet(spikes, filepath = output_sequences_path)
write.csv(metadata, row.names = FALSE, file = output_metadata_path)

