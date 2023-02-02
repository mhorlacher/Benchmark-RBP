#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import argparse
import tempfile

import sys
from pathlib import Path


# In[6]:


def sample_from_bed(bed_filepaths, n):
    total = 0
    for bed in bed_filepaths:
        with open(bed) as f:
            for _ in f:
                total += 1
    
    if n > total:
        raise ValueError(f'Can not draw {n} sampled from BED file of length {total}.')
    
    nsample = np.random.choice(np.sum(total, dtype=np.uint32) - 1, size=n, replace=False)
    nsample = set(nsample)
    
    for bed in bed_filepaths:
        with open(bed) as f:
            for i, line in enumerate(f):
                if i in nsample:
                    print(line.strip())


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, help='total number of rows to sample')
    parser.add_argument('in_beds', metavar='<in.bed>', nargs='+')
    args = parser.parse_args()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_from_bed(args.in_beds, args.n)

