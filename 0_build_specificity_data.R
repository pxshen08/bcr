## ---------------------------
## Script name: 5_build_specificity_data.R
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
})

sequence = "CDR3"

if (sequence == "CDR3") {
    cols_1 = c("junction_aa_heavy", "junction_aa_light", "elisa", "donor")
    cols_2 = c("CDRH3", "CDRL3", "label", "Sources")
} else if (sequence == "FULL") {
    cols_1 = c("sequence_alignment_aa_heavy", "sequence_alignment_aa_light", "elisa", "donor")
    cols_2 = c("VHorVHH", "VL", "label", "Sources")
}    
cols_std = c("heavy", "light", "label", "group")

qc_seq = function(df) {
  df %>%
      mutate(heavy = gsub(" |X", "",heavy)) %>%
      filter(heavy != "ND") %>%
      filter(!grepl("*", heavy, fixed = T)) %>%
      filter(!is.na(heavy)) %>%
      mutate(light = gsub(" |X", "",light))
}

##-----------------------------
## Load training data

covabdb = read_csv("/gpfs/ysm/project/kleinstein/mw957/repos/bcr_embeddings/data/CoV-AbDab_260722.csv")
covabdb = covabdb %>%
  filter((`Doesn't Bind to` == "SARS-CoV2_WT" ) | `Binds to` == "SARS-CoV2_WT") %>%
  filter(!grepl("Wooseob", `Sources`))
covabdb$label = NA
covabdb$label[covabdb$`Doesn't Bind to`== "SARS-CoV2_WT"] = F
covabdb$label[covabdb$`Binds to`== "SARS-CoV2_WT"] = T

train = covabdb[,cols_2]
colnames(train) = cols_std



##-----------------------------
## Load test data

basedir = "/gpfs/gibbs/pi/kleinstein/ellebedy_COVID19"
metadata = read_tsv(file.path(basedir, "/meta_47_share.tsv"))
ellebedy_covid19 = read_tsv("/gpfs/gibbs/pi/kleinstein/ellebedy_COVID19/out_VDJ/6_translate/filtered_translated.tsv")

ellebedy_covid19 = ellebedy_covid19 %>%
  filter(!is.na(elisa)) %>%
  left_join(metadata[,c("donor", "sample")]) 

test = ellebedy_covid19[,cols_1]
colnames(test) = cols_std

if(sequence == "CDR3") {
    test$heavy = map_chr(test$heavy, ~str_sub(.x, 2, -2))
    test$light = map_chr(test$light, ~str_sub(.x, 2, -2))
}

##-----------------------------
## QC sequences

train = qc_seq(train)

test = qc_seq(test)

table(train$label) 
# FULL
# FALSE  TRUE 
#  198  3946
# CDR3
# FALSE  TRUE 
#   182  3588

table(test$label) 
# FALSE  TRUE 
#   545  1440

##-----------------------------
## exclude the ovelapping test sequences from train
train = train %>%
  left_join(test[,1:2] %>% mutate(in_test = "Yes")) %>%
  filter(is.na(in_test)) %>%
  select(-in_test)

##-----------------------------
# Add in random negatives from Shaw flu datasets
# to balance the class
train_controls = read_tsv("/gpfs/gibbs/pi/kleinstein/mw957/BCR_embed/data/shaw_flu/filtered_translated.tsv", col_types = cols()) %>%
  filter(grepl("_0", sample)) %>%
  group_by(sample, clone_id_heavy) %>%
  sample_n(size = 1) %>%
  ungroup()
train_controls = train_controls[,c(cols_1[1:2],"sample")]
train_controls = distinct(train_controls) 
colnames(train_controls)[1:2] = c("heavy", "light")

train_controls = train_controls %>%
  left_join(rbind(train, test)[,1:2] %>% mutate(appeared = "Yes")) %>%
  filter(is.na(appeared)) %>%
  select(-appeared)

set.seed(10)
train_controls = train_controls %>%
  group_by(sample) %>%
  sample_n(600) %>%
  mutate(label = F) %>%
  rename(group = sample)

if(sequence == "CDR3") {
    train_controls$heavy = map_chr(train_controls$heavy, ~str_sub(.x, 2, -2))
    train_controls$light = map_chr(train_controls$light, ~str_sub(.x, 2, -2))
}

train = bind_rows(train, train_controls)

##-----------------------------
# get the corresponding index number from the fasta file
basedir = "/gpfs/ysm/project/kleinstein/mw957/repos/bcr_embeddings/data/"
if(sequence == "FULL") {
  heavy_name = "combined_distinct_heavy.csv"
  light_name = "combined_distinct_light.csv"
  out_name_train = "specificity.anno"
  out_name_test = "specificity_test.anno"
  
} else if (sequence == "CDR3") {
  heavy_name = "combined_cdr3_heavy.csv"
  light_name = "combined_cdr3_light.csv"
  out_name_train = "cdr3_specificity.anno"
  out_name_test = "cdr3_specificity_test.anno"
}
heavy_id = read_csv(file.path(basedir, heavy_name)) 
colnames(heavy_id)[1] = "heavy"
light_id = read_csv(file.path(basedir, light_name))
colnames(light_id)[1] = "light"

train = train %>%
  left_join(heavy_id) %>%
  left_join(light_id)
test = test %>%
  left_join(heavy_id) %>%
  left_join(light_id)

## ---------------------------
## Save distinct heavy and light chain files

write_tsv(train[,c(5,6,3,4)], file.path("/gpfs/gibbs/pi/kleinstein/mw957/BCR_embed/data/", out_name_train))
cat("Writing ", nrow(train), "training examples.\n")
table(train$label) 
# FULL
# FALSE  TRUE 
#  3782  3588
# CDR3
# FALSE  TRUE 
# 3798  3939

cat("Writing ", nrow(test), "test examples.\n")
table(test$label) 
# FALSE  TRUE 
#   545  1440

write_tsv(test[,c(5,6,3,4)], file.path("/gpfs/gibbs/pi/kleinstein/mw957/BCR_embed/data/", out_name_test))
## ---------------------------