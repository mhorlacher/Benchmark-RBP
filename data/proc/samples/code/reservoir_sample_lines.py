#!/usr/bin/env python
# coding: utf-8

# In[3]:


import argparse
import random


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument('file', metavar='<file>')
parser.add_argument('-n', '--number', type=int)
parser.add_argument('-l', '--lines-per-sample', type=int)
parser.add_argument('--skip-header', type=int, default=0)
args = parser.parse_args()


# In[2]:


def read_file(file):
    sample = ''
    
    with open(file) as f:
        for _ in range(0, args.skip_header):
            _ = f.readline()

        for i, line in enumerate(f, start=1):
            sample += line
            if i % args.lines_per_sample == 0:
                yield sample
                sample = ''


# In[1]:


def reservoir_sample( iterator, K ):
    result = []
    N = 0

    for item in iterator:
        N += 1
        if len( result ) < K:
            result.append( item )
        else:
            s = int(random.random() * N)
            if s < K:
                result[ s ] = item

    return result


# In[ ]:


if __name__ == '__main__':
    for sample in reservoir_sample(read_file(args.file), args.number):
        print(sample, end='')

