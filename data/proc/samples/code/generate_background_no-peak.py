#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Generate negative background sequences by sampling from peak-containing transcripts outside of peak regions."""


# In[ ]:


import sys
import copy


# In[ ]:


import argparse
import subprocess
import tempfile
from pathlib import Path


# In[ ]:


import numpy as np


# In[ ]:


def get_transcripts_with_peaks(peaks_bed, transcripts_bed, out_bed=None):
    """Select all transcripts which overlap with at least one peak window and write to out_bed."""
    
    if out_bed is None:
        out_bed = transcripts_bed.strip('.bed') + '.overlapping-peaks.bed'
    
    subprocess.run(f'bedtools intersect -s -wa -u -a {transcripts_bed} -b {peaks_bed} > {out_bed}', shell=True)
    
    return out_bed


# In[ ]:


def substract_overlapping_regions(in_bed, substract_bed, out_bed=None):
    """Remove peak windows from transcripts and write to out_bed."""
    if out_bed is None:
        out_bed = in_bed.strip('.bed') + '.substracted.bed'
    
    # bedtools: substract 
    subprocess.run(f'bedtools subtract -s -u -wa -a {in_bed} -b {substract_bed} > {out_bed}', shell=True, capture_output=True)
        
    return out_bed


# In[ ]:


def sort_bed(in_bed, out_bed=None):
    """Sort a bed-file and write sorted file to out_bed."""
    if out_bed is None:
        out_bed = in_bed.strip('.bed') + '.sorted.bed'
    
    # sort bed by chrom/start
    subprocess.run(f'sort -k1,1 -k2,2n {in_bed} > {out_bed}', shell=True)
    
    return out_bed


# In[ ]:


def merge_bed(in_bed, out_bed=None):
    """Merge overlapping regions of a bed-file."""
    if out_bed is None:
        out_bed = in_bed.strip('.bed') + '.merged.bed'
    
    # bedtools: merge overlapping ranges of bed
    subprocess.run((f"bedtools merge -s -c 4,5,6 -o collapse,distinct,distinct -delim ';' -i {in_bed} > {out_bed}"), shell=True)
    
    return out_bed


# In[2]:


def row_key(row):
    return f'{row[0]}_{row[1]}'


# In[ ]:


def sample_windows_from_bed(in_bed, n, window_size=400):
    """Sample windows of size window_size from all ranges of a bed-file and print to stdout."""
    
    # Build a index/key list for all possible ranges
    in_bed_row_tuples = list()
    in_bed_row_sizes = list()
    total_size = 0

    with open(in_bed) as f:
        for line in f:
            row = line.strip().split('\t')
            
            row_size = int(row[2]) - int(row[1]) - window_size
            if row_size > 0:
                in_bed_row_tuples.append(row)
                in_bed_row_sizes.append(row_size)
                total_size += row_size
                
    if args.verbose:
        print(f'Total sequences to sample: {len(in_bed_row_sizes)}', file=sys.stderr)
        print(f'Total positions to sample: {sum(in_bed_row_sizes)}', file=sys.stderr)

    p = np.array(in_bed_row_sizes, dtype=np.float64) / total_size
    
    sampled_row_ids = set()
    duplicates = 0
    sampled = 0
    while sampled < n:
        row_sample = copy.copy(in_bed_row_tuples[np.random.choice(len(in_bed_row_tuples), p=p)])
        start_idx_sample = np.random.choice(int(row_sample[2]) - int(row_sample[1]) - window_size + 1)
        
        start = int(row_sample[1])
        
        row_sample[1] = str(start + start_idx_sample)
        row_sample[2] = str(start + start_idx_sample + window_size)
        
        assert int(row_sample[2]) - int(row_sample[1]) == window_size
        
        if row_key(row_sample) in sampled_row_ids:
            if args.resample_on_duplicate:
                continue
            else:
                duplicates += 1
                
        print('\t'.join(row_sample))

        sampled_row_ids.add(row_key(row_sample))
        sampled += 1
        
    if args.verbose:
        print(f'Sampled {n} windows and encountered {duplicates} duplicates ({len(sampled_row_ids)} unique)', file=sys.stderr)


# In[ ]:


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('peaks_bed', metavar='<peaks.bed>')
    parser.add_argument('transcripts_bed', metavar='<transcripts.bed>')
    parser.add_argument('-n', '--number', type=int, help='Number of windows to sample.')
    parser.add_argument('-s', '--seed', type=int, help='Seed to use for sampling')
    parser.add_argument('-w', '--window-size', type=int, default = 400)
    parser.add_argument('--resample-on-duplicate', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ov_trans = get_transcripts_with_peaks(args.peaks_bed, args.transcripts_bed, out_bed = str(Path(tmpdir).joinpath('ov_trans.bed')))
        ov_trans_sub = substract_overlapping_regions(ov_trans, args.peaks_bed, out_bed = str(Path(tmpdir).joinpath('ov_trans_sub.bed')))
        ov_trans_sub_sort = sort_bed(ov_trans_sub, out_bed = str(Path(tmpdir).joinpath('ov_trans_sub_sort.bed')))
        ov_trans_sub_sort_merg = merge_bed(ov_trans_sub_sort, out_bed = str(Path(tmpdir).joinpath('ov_trans_sub_sort_merg.bed')))
    
        overlapping_transcripts_substracted_sorted_sample = sample_windows_from_bed(ov_trans_sub_sort_merg, n=args.number, window_size = args.window_size)

