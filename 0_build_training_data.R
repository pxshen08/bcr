## ---------------------------
## Script name: 0_combine_data.R
##
## Purpose of script: Combine all available datasets together
##
## Author: Mamie Wang
##
## Date Created: 2022-09-22
##
## Email: mamie.wang@yale.edu
## ---------------------------

## load the packages and functions
suppressPackageStartupMessages({
    library(tidyverse)
    library(seqinr)
})

find.characters <- function(v1){
    x1 <- unique(unlist(strsplit(v1, '')))
    indx <- grepl('[A-Z]', x1)
    c(sort(x1[indx]), sort(x1[!indx]))
}

##-----------------------------
## Load data

train_1 = read_tsv("/gpfs/gibbs/pi/kleinstein/ellebedy_flu/BCR/02_translated/filtered_translated.tsv", col_types = cols()) %>% mutate(data = "Ellebedy flu", repertoire_id = as.character(sample_index))
train_2 = read_tsv("/gpfs/gibbs/pi/kleinstein/ellebedy_COVID19/out_VDJ/6_translate/filtered_translated.tsv", col_types = cols())  %>% mutate(data = "Ellebedy COVID19", repertoire_id = sample)
train_3 = read_tsv("/gpfs/gibbs/pi/kleinstein/mw957/BCR_embed/data/shaw_flu/filtered_translated.tsv", col_types = cols())  %>% mutate(data = "Shaw flu", repertoire_id = sample)
train_4 = read_csv("/gpfs/ysm/project/kleinstein/mw957/repos/bcr_embeddings/data/CoV-AbDab_260722.csv", col_types = cols())  %>% mutate(data = "CoV-AbDab", repertoire_id = "CoV-AbDab")
train_5 = read_tsv("/gpfs/gibbs/pi/kleinstein/mw957/data/oas_paired/data_wide.tsv", col_types = cols())  %>% mutate(data = "OAS", repertoire_id = sample_id)
train_6 = read_tsv("/gpfs/ysm/project/mw957/repos/bukreyev_covid/data/data_wide.tsv", col_types = cols())  %>% mutate(data = "Bukreyev COVID19", repertoire_id = sample)
train_7 = read_tsv("/gpfs/ysm/project/mw957/repos/kaminsky_covid/data/data_wide.tsv", col_types = cols())  %>% mutate(data = "Kaminski COVID19", repertoire_id = sample_id)
train_8 = read_tsv("/gpfs/ysm/project/mw957/repos/ireceptor_plus/data/ir_paired/data_wide.tsv", col_types = cols()) %>% mutate(data = "iR+ COVID19", repertoire_id = sample_id)
train_9 = read_tsv("/gpfs/gibbs/pi/kleinstein/mw957/data/moir_covid/data_wide.tsv", col_types = cols()) %>%
  mutate(data = "Moir COVID19", repertoire_id = sample_id)
train_10 = read_tsv("/gpfs/gibbs/pi/kleinstein/mw957/data/schwartzberg_covid/data_wide.tsv", col_types = cols()) %>%
  mutate(data = "Schwartzberg COVID19", repertoire_id = donor)

data_levels = c("Ellebedy flu", "Shaw flu", "Ellebedy COVID19", "Bukreyev COVID19",
                "Kaminski COVID19", "Moir COVID19", "Schwartzberg COVID19", 
                "CoV-AbDab", "iR+ COVID19", "OAS")

cols_1 = c("sequence_alignment_aa_heavy", "sequence_alignment_aa_light", "data")
cols_2 = c("VHorVHH", "VL")

train_4_seq = train_4[,cols_2]
colnames(train_4_seq) = cols_1

train = bind_rows(train_1[,cols_1] ,
                  train_2[,cols_1],
                  train_3[,cols_1],
                  train_4_seq,
                  train_5[,cols_1],
                  train_6[,cols_1],
                  train_7[,cols_1],
                  train_8[,cols_1],
                  train_9[,cols_1],
                  train_10[,cols_1]) %>%
  mutate(data = factor(data, levels = data_levels))

## ---------------------------
## Save distinct heavy files
distinct_heavy = train %>% 
  mutate(sequence_alignment_aa_heavy = gsub(" |X", "",sequence_alignment_aa_heavy)) %>%
  filter(sequence_alignment_aa_heavy != "ND") %>%
  filter(!grepl("*", sequence_alignment_aa_heavy, fixed = T)) %>%
  distinct(sequence_alignment_aa_heavy)

print(find.characters(distinct_heavy$sequence_alignment_aa_heavy))
cat("Found ", nrow(distinct_heavy), " distinct heavy chains.\n") # 865947
write.fasta(as.list(distinct_heavy$sequence_alignment_aa_heavy), 1:nrow(distinct_heavy), "/gpfs/ysm/project/kleinstein/mw957/repos/bcr_embeddings/data/combined_distinct_heavy.fa")
write_csv(distinct_heavy %>% mutate(heavy_id = 1:nrow(distinct_heavy)), "/gpfs/ysm/project/kleinstein/mw957/repos/bcr_embeddings/data/combined_distinct_heavy.csv")

## ---------------------------
## Save distinct heavy metadata files
distinct_heavy = distinct_heavy %>%
  mutate(id = 0:(nrow(distinct_heavy)-1))

cols_1 = c("sequence_alignment_aa_heavy", 
           "v_call_gene_heavy", 
           "v_call_family_heavy", 
           "j_call_family_heavy",
           "mu_freq_heavy",
           "junction_aa_length_heavy", 
           "isotype_heavy",
           "data")
gene_call = bind_rows(train_1[,cols_1],
                  train_2[,cols_1],
                  train_3[,cols_1],
                  train_5[,cols_1],
                  train_6[,cols_1],
                  train_7[,cols_1],
                  train_8[,cols_1],
                  train_9[,cols_1],
                  train_10[,cols_1]) %>%
  mutate(sequence_alignment_aa_heavy = gsub(" |X", "",sequence_alignment_aa_heavy)) %>%
  filter(sequence_alignment_aa_heavy != "ND") %>%
  filter(!grepl("*", sequence_alignment_aa_heavy, fixed = T)) %>%
  distinct %>%
  inner_join(distinct_heavy)

gene_call = gene_call[!duplicated(gene_call$sequence_alignment_aa_heavy),]
gene_call$sequence_length_heavy = nchar(gene_call$sequence_alignment_aa_heavy)
# randomly resolve different labels
write_tsv(gene_call[,-1], "/gpfs/gibbs/pi/kleinstein/mw957/BCR_embed/data/combined_distinct_heavy.anno")
cat("Wrote ", nrow(gene_call), " distinct heavy chains annotations.\n") # 858682

## ---------------------------
## Save distinct light files

distinct_light = train %>% 
  mutate(sequence_alignment_aa_light = gsub(" |X", "",sequence_alignment_aa_light)) %>%
  filter(sequence_alignment_aa_light != "ND") %>%
  filter(!grepl("*", sequence_alignment_aa_light, fixed = T)) %>%
  distinct(sequence_alignment_aa_light)

print(find.characters(distinct_light$sequence_alignment_aa_light))
cat("Found ", nrow(distinct_light), " distinct light chains.\n") # 546581

write.fasta(as.list(distinct_light$sequence_alignment_aa_light), 1:nrow(distinct_light), "/gpfs/ysm/project/kleinstein/mw957/repos/bcr_embeddings/data/combined_distinct_light.fa")
write_csv(distinct_light %>% mutate(light_id = 1:nrow(distinct_light)), "/gpfs/ysm/project/kleinstein/mw957/repos/bcr_embeddings/data/combined_distinct_light.csv")
## ---------------------------
## Save distinct light metadata files

distinct_light = distinct_light %>%
  mutate(id = 0:(nrow(distinct_light)-1))

cols_1 = c("sequence_alignment_aa_light", 
           "v_call_gene_light", "v_call_family_light",
           "j_call_family_light",
           "mu_freq_light",
           "junction_aa_length_light", 
           "isotype_light", "data")
gene_call = bind_rows(train_1[,cols_1],
                  train_2[,cols_1],
                  train_3[,cols_1],
                  train_5[,cols_1],
                  train_6[,cols_1],
                  train_7[,cols_1],
                  train_8[,cols_1],
                  train_9[,cols_1],
                  train_10[,cols_1]) %>%
  mutate(sequence_alignment_aa_light = gsub(" |X", "",sequence_alignment_aa_light)) %>%
  filter(sequence_alignment_aa_light != "ND") %>%
  filter(!grepl("*", sequence_alignment_aa_light, fixed = T)) %>%
  distinct %>%
  inner_join(distinct_light)
gene_call = gene_call[!duplicated(gene_call$sequence_alignment_aa_light),]
gene_call$sequence_length_light = nchar(gene_call$sequence_alignment_aa_light)
write_tsv(gene_call[,-1], "/gpfs/gibbs/pi/kleinstein/mw957/BCR_embed/data/combined_distinct_light.anno")
cat("Wrote ", nrow(gene_call), " distinct light chains annotations.\n") # 540684
## ---------------------------
## Save distinct heavy and light chain files for CDR3
cols_1 = c("junction_aa_heavy", "junction_aa_light")
cols_2 = c("CDRH3", "CDRL3")

train_4_seq = train_4[,cols_2]
colnames(train_4_seq) = cols_1

train = bind_rows(train_1[,cols_1] %>% mutate(data = "Ellebedy flu", repertoire_id = as.character(train_1$sample_index)),
                  train_2[,cols_1] %>% mutate(data = "Ellebedy COVID19", repertoire_id = train_2$sample),
                  train_3[,cols_1] %>% mutate(data = "Shaw flu", repertoire_id = train_3$sample),
                  train_4_seq %>% mutate(data = "CoV-AbDab", repertoire_id = "CoV-AbDab"),
                  train_5[,cols_1] %>% mutate(data = "OAS", repertoire_id = train_5$sample_id),
                  train_6[,cols_1] %>% mutate(data = "Bukreyev COVID19", repertoire_id = train_6$sample),
                  train_7[,cols_1] %>% mutate(data = "Kaminski COVID19", repertoire_id = train_7$sample_id),
                  train_8[,cols_1] %>% mutate(data = "iR+ COVID19", repertoire_id = train_8$sample_id),
                  train_9[,cols_1] %>% mutate(data = "Moir COVID19", repertoire_id = as.character(train_9$sample_id)),
                  train_10[,cols_1] %>% mutate(data = "Schwartzberg COVID19", repertoire_id = train_10$donor)) %>%
  mutate(data = factor(data, levels = data_levels))

trim <- function(strings) {
    str_sub(strings, 2, -2)
}

train$junction_aa_heavy[train$data != "CoV-AbDab"] = trim(train$junction_aa_heavy[train$data != "CoV-AbDab"])

train$junction_aa_light[train$data != "CoV-AbDab"] = trim(train$junction_aa_light[train$data != "CoV-AbDab"])

distinct_heavy = train %>% 
  mutate(junction_aa_heavy = gsub(" |X", "",junction_aa_heavy)) %>%
  filter(!junction_aa_heavy %in% c("ND", "")) %>%
  filter(!grepl("*", junction_aa_heavy, fixed = T)) %>%
  distinct(junction_aa_heavy) %>%
  filter(nchar(junction_aa_heavy) > 2)

failed = which(grepl("[0-9]", train$junction_aa_light))

distinct_light = train[-failed,] %>% 
  mutate(junction_aa_light = gsub(" |X", "",junction_aa_light)) %>%
  filter(!junction_aa_light %in% c("ND", "")) %>%
  filter(!grepl("*", junction_aa_light, fixed = T)) %>%
  distinct(junction_aa_light)  %>%
  filter(nchar(junction_aa_light) > 2)

find.characters <- function(v1){
    x1 <- unique(unlist(strsplit(v1, '')))
    indx <- grepl('[A-Z]', x1)
    c(sort(x1[indx]), sort(x1[!indx]))
}
find.characters(distinct_heavy$junction_aa_heavy)

find.characters(distinct_light$junction_aa_light)

cat("Found ", nrow(distinct_heavy), " distinct CDR3 heavy chains.\n") # 791890
cat("Found ", nrow(distinct_light), " distinct CDR3 light chains.\n") # 223923

write.fasta(as.list(distinct_heavy$junction_aa_heavy), 1:nrow(distinct_heavy), "/gpfs/ysm/project/kleinstein/mw957/repos/bcr_embeddings/data/combined_cdr3_heavy.fa")
write_csv(distinct_heavy %>% mutate(heavy_id = 1:nrow(distinct_heavy)), "/gpfs/ysm/project/kleinstein/mw957/repos/bcr_embeddings/data/combined_cdr3_heavy.csv")

write.fasta(as.list(distinct_light$junction_aa_light), 1:nrow(distinct_light), "/gpfs/ysm/project/kleinstein/mw957/repos/bcr_embeddings/data/combined_cdr3_light.fa")
write_csv(distinct_light %>% mutate(light_id = 1:nrow(distinct_light)), "/gpfs/ysm/project/kleinstein/mw957/repos/bcr_embeddings/data/combined_cdr3_light.csv")

## ---------------------------