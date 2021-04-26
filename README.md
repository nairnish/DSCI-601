# DSCI-601

Discriminating between similar languages

This repository consists of several experimentation carried out on the DSLCC dataset v4.0 for discriminating between similar language variants.

Below are the list of items present in each folder:

1. **601 project experiments - 1** (contains experiments with tf-idf word uni/bi grams)
    -> Codes - Contains the model classification source codes and associated testing codes
    -> DSLCC4 datastes - Contains the required datasets for the model to train, validate and test
    -> Evaluation - Contains the code for evaluation of the resuls from model classification

2. **data** - Contains all the necessary datasets required for the experiments
3. **doc - 1** - Contains the code documentation generated by doxygen for the experiments

Information on the Dataset:
DSLCC 4.0 Corpus
================

This is the training and test data for the Discriminating between Similar Languages (DSL) task at VarDial 2017.

The package contains the following files:

DSL-TRAIN.txt 						- Training set for the DSL task
DSL-DEV.txt 							- Development set for the DSL task
DSL-DEV.txt 							- Unlabelled test set
DSL-TEST-UNLABELLED.txt 	- Test set with gold labels
README.txt 								- Brief description of the DSL data

Each line in the .txt files are tab-delimited in the format:
sentence<tab>language-label

For more details (like data stats) you can refer to the VarDial 2017 task paper:

Marcos Zampieri, Shervin Malmasi, Nikola Ljubesic,
Preslav Nakov, Ahmed Ali, Jorg Tiedemann, Yves
Scherrer, and Noemi Aepli. 2017. "Findings of the
VarDial Evaluation Campaign 2017." In Proceedings
of the Fourth Workshop on NLP for Similar Languages,
Varieties and Dialects (VarDial), Valencia, Spain.
