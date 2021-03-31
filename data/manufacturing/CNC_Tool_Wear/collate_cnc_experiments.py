#!/usr/bin/env python
#
###########################################
#
# File: collate_cnc_experiments.py
# Author: Ra Inta
# Description: Collation script to combine information
# on experiments from University of Michigan's CNC
# experiments. There is a CSV file with labels
# (train.csv) that are controlled variables
# for each of the 18 files of other data.
# This script combines these labels with
# the other data into a single file to rule
# them all.
#
# Created: May 14, 2019
# Last Modified: May 14, 2019
#
###########################################

import pandas as pd

data_labels = pd.read_csv("train.csv")

file_num = data_labels.pop('No').to_list()

new_col_names = data_labels.columns.values

full_df = pd.DataFrame()

for fileIdx in file_num:
    current_df = pd.read_csv(f"experiment_{fileIdx:0>2}.csv")  # _have_ to love f-strings
    # Exploit broadcasting for new columns
    for new_col in new_col_names:
        current_df[new_col] = data_labels.loc[fileIdx - 1, new_col]
    full_df = full_df.append(current_df)

full_df.to_csv("cnc_experiments_all.csv", index=False)

###########################################
# End of collate_cnc_experiments.py
###########################################
