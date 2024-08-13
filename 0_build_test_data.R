## ---------------------------
## Script name: 5_build_task_data.R
##
## Purpose of script: Combine all available datasets together
##
## Author: Mamie Wang
##
## Date Created: 2022-09-23
##
## Input: BCR data with annotation, and embedding file
## Output: 2 Pickled data frame with embeddings and labels for the tests
## - labels:
##   - v gene, j gene, mu_freq, cdr3 length 
## Email: mamie.wang@yale.edu
## ---------------------------

## load the packages and functions
suppressPackageStartupMessages({
    library(tidyverse)
    library(seqinr)
})

##-----------------------------
## Load data

base_dir = "/home/mw957/project/repos/meffre_CTLA4/data/human/B_cells/out_VDJ/"
data = read_tsv(file.path(base_dir, "data_wide.tsv"))

cols_1 = c("sequence_alignment_aa_heavy", "sequence_alignment_aa_light")

test = data[,cols_1]

## ---------------------------
## Save distinct heavy and light chain files

distinct_heavy = test %>% 
  mutate(sequence_alignment_aa_heavy = gsub(" |X", "",sequence_alignment_aa_heavy)) %>%
  filter(sequence_alignment_aa_heavy != "ND") %>%
  filter(!grepl("*", sequence_alignment_aa_heavy, fixed = T)) %>%
  distinct(sequence_alignment_aa_heavy)

distinct_light = test %>% 
  mutate(sequence_alignment_aa_light = gsub(" |X", "",sequence_alignment_aa_light)) %>%
  filter(sequence_alignment_aa_light != "ND") %>%
  filter(!grepl("*", sequence_alignment_aa_light, fixed = T)) %>%
  distinct(sequence_alignment_aa_light)

find.characters <- function(v1){
    x1 <- unique(unlist(strsplit(v1, '')))
    indx <- grepl('[A-Z]', x1)
    c(sort(x1[indx]), sort(x1[!indx]))
}

print(find.characters(distinct_heavy$sequence_alignment_aa_heavy))

print(find.characters(distinct_light$sequence_alignment_aa_light))

write.fasta(as.list(distinct_heavy$sequence_alignment_aa_heavy), 1:nrow(distinct_heavy), "/gpfs/ysm/project/kleinstein/mw957/repos/bcr_embeddings/data/combined_distinct_heavy_test.fa")

write.fasta(as.list(distinct_light$sequence_alignment_aa_light), 1:nrow(distinct_light), "/gpfs/ysm/project/kleinstein/mw957/repos/bcr_embeddings/data/combined_distinct_light_test.fa")

## ---------------------------
## Save annotations for test dataset
distinct_heavy = distinct_heavy %>%
  mutate(id = 0:(nrow(distinct_heavy)-1))

distinct_light = distinct_light %>%
  mutate(id = 0:(nrow(distinct_light)-1))

cols_1 = c("sequence_alignment_aa_heavy", 
           "v_call_gene_heavy", 
           "v_call_family_heavy",
           "j_call_family_heavy",
           "mu_freq_heavy",
           "isotype_heavy",
           "junction_aa_length_heavy")

gene_call = data[, cols_1] %>%
     mutate(sequence_alignment_aa_heavy = gsub(" |X", "",sequence_alignment_aa_heavy)) %>%
  filter(sequence_alignment_aa_heavy != "ND") %>%
  filter(!grepl("*", sequence_alignment_aa_heavy, fixed = T)) %>%
  distinct %>%
  inner_join(distinct_heavy)

print(sum(duplicated(gene_call$sequence_alignment_aa_heavy)))

gene_call = gene_call[!duplicated(gene_call$sequence_alignment_aa_heavy),]
gene_call$sequence_length_heavy = nchar(gene_call$sequence_alignment_aa_heavy)
write_tsv(gene_call[,-1], "/gpfs/gibbs/pi/kleinstein/mw957/BCR_embed/data/combined_distinct_heavy_test.anno")

# Light chain

cols_1 = c("sequence_alignment_aa_light", 
           "v_call_gene_light", 
           "v_call_family_light",
           "j_call_family_light",
           "mu_freq_light",
           "isotype_light",
           "junction_aa_length_light")
gene_call = data[,cols_1] %>%
  mutate(sequence_alignment_aa_light = gsub(" |X", "",sequence_alignment_aa_light)) %>%
  filter(sequence_alignment_aa_light != "ND") %>%
  filter(!grepl("*", sequence_alignment_aa_light, fixed = T)) %>%
  distinct %>%
  inner_join(distinct_light)

print(sum(duplicated(gene_call$sequence_alignment_aa_light)))

gene_call = gene_call[!duplicated(gene_call$sequence_alignment_aa_light),]
gene_call$sequence_length_light = nchar(gene_call$sequence_alignment_aa_light)
write_tsv(gene_call[,-1], 
          "/gpfs/gibbs/pi/kleinstein/mw957/BCR_embed/data/combined_distinct_light_test.anno")

## ---------------------------


