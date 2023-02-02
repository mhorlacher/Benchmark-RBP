#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import argparse
import subprocess
import tempfile
from pathlib import Path


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument('bed_1', metavar = '1.bed')
parser.add_argument('bed_2', metavar = '2.bed')
parser.add_argument('-f', type=float, default=None)
parser.add_argument('-u', action='store_true', default=False)
args = parser.parse_args()


# In[ ]:


with tempfile.TemporaryDirectory() as tmpdir:
    cmd = f'bedtools intersect -f {args.f if args.f is not None else ""} {"-u" if args.u else ""} -s -wa -a {args.bed_1} -b {args.bed_2} > {str(Path(tmpdir) / "1_2.isec.bed")}'
    print(cmd, file=sys.stderr)
    subprocess.run(cmd, shell=True)
    
    cmd = f'bedtools intersect -f {args.f if args.f is not None else ""} {"-u" if args.u else ""} -s -wa -a {args.bed_2} -b {args.bed_1} > {str(Path(tmpdir) / "2_1.isec.bed")}'
    print(cmd, file=sys.stderr)
    subprocess.run(cmd, shell=True)
    
    with open(str(Path(tmpdir) / "1_2.isec.bed")) as f:
        for line in f:
            print(line.strip())

    with open(str(Path(tmpdir) / "2_1.isec.bed")) as f:
        for line in f:
            print(line.strip())

