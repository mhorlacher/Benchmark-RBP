#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse


# In[1]:


import pandas as pd


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, help='Maximum number of rows to select.')
parser.add_argument('--min-FC', type=float, help='Minimum fold-chage versus control.')
parser.add_argument('peaks_bed', metavar='<peaks.bed>')
parser.add_argument('peaks_top_bed', metavar='<peaks.top.bed>')
args = parser.parse_args()


# In[ ]:


peaks_df = pd.read_csv(args.peaks_bed, delimiter='\t', header=None)
peaks_df = peaks_df.sort_values(4, ascending=False) # Sort values by FC
peaks_df = peaks_df.loc[peaks_df[4] >= args.min_FC] # Apply FC cutoff
peaks_df = peaks_df.head(args.n) # Subset top n rows


# In[ ]:


peaks_df.to_csv(args.peaks_top_bed, sep='\t', header=False, index=False)

