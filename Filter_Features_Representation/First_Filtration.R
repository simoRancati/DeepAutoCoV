library("Biostrings")
library(stringr)
library(vroom)

setwd('') # Set the directory containing folders of metadata file (metadata.tsv) and sequences (spikes.fasta)

spikes = readAAStringSet('') # Set the folder containing the fasta file 

original_spike_names = names(spikes)
names(spikes) = unlist(lapply(str_split(names(spikes), '\\|'), '[[', 4))

metadata <- vroom("metadata/metadata.tsv") # Read metadata file (metadata.tsv)
spec(metadata)

table(metadata$Variant)
                                                                              
# keep only high q + complete
metadata = metadata[metadata$`Is high coverage?`==TRUE &
                      !is.na(metadata$`Is high coverage?`) & 
                      metadata$`Is complete?`==TRUE &
                      metadata$Host == "Human" &
                      !is.na(metadata$`Is complete?`) &
                      !is.na(metadata$`Pango lineage`) &
                      !is.na(metadata$`Submission date`) & # we use the real submission data as we are not simulating to have the sequences right away
                      str_length(metadata$`Submission date`)==10,]

# exclude incongruent instances not present in one of the two files
spikes = spikes[names(spikes) %in% metadata$`Accession ID`] 
metadata = metadata[metadata$`Accession ID` %in% names(spikes),] 

writeXStringSet(spikes,
                filepath = paste0("")) # Save the filtered sequences in Fasta file 
write.csv(metadata, row.names = FALSE, paste0("")) # Save the filtered metadata in csv file 

