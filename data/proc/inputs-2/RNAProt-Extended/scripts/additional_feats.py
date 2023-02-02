import os, argparse
import subprocess
import pickle
import pandas as pd

from rnaprot.rplib import bed_get_exon_intron_annotations_from_gtf, extract_exon_intron_labels, gtf_get_transcript_ids, gtf_extract_most_prominent_transcripts

# Transcript IDs dictionary
# Test tr_ids_dic = {'tr1': 1, 'tr2': 1}  - has to be transcript_id -> chromosome
tr_ids_dic = {}
# test_eia.bed
# chr1	1000	1005	reg1	0	+
# chr1	1020	1025	reg2	0	+
in_bed = "../../RNAProt/processed/Mukherjee-PAR-CLIP/AGO1_HEK293_PARCLIP/fold-0/positive.fold-0.bed"

# test_eia.gtf - ensembl annotation format
# 1	havana	transcript	101	200	.	+	.	gene_id "gene1"; transcript_id "tr1";
# 1	havana	exon	101	120	.	+	.	gene_id "gene1"; transcript_id "tr1"; exon_id "ex1"; exon_number "1";
in_gtf = "/lustre/groups/crna01/workspace/giulia/projects/Benchmark-RBP/data/meta/ensembl/hg19/Homo_sapiens.GRCh37.87.gtf"

###### OUT FILES 

in_bed_with_ids = "/lustre/groups/crna01/projects/Benchmark-RBP/data/proc/inputs-2/RNAProt-Extended/test_in_bed_ids.out"
# Example feat_bed
# chr1	1002	1021	feat1	5	+
# chr1	1000	2000	feat2	10	-
# feat_bed = "/lustre/groups/crna01/projects/Benchmark-RBP/data/proc/inputs-2/RNAProt-Extended/test_transcript.out"

out_file = "/lustre/groups/crna01/projects/Benchmark-RBP/data/proc/inputs-2/RNAProt-Extended/test_overlap.out"
out_tra = "/lustre/groups/crna01/projects/Benchmark-RBP/data/proc/inputs-2/RNAProt-Extended/test_anno_tra.out"
eia_out = "/lustre/groups/crna01/projects/Benchmark-RBP/data/proc/inputs-2/RNAProt-Extended/test_eia.out"

# gtf_extract_transcript_bed(in_gtf, feat_bed, tr_ids_dic=False)

# Add unique ids in col 'name' (4) of input bed - required for all functions in rplib.py
# TODO Could not find a function in rnaprot.rplib that does it?

in_bed_df = pd.read_csv(in_bed, sep="\t", names=["chr", "start", "end", "name", "score", "strand"])
in_bed_df['name'] = in_bed_df.apply(lambda x: f"{x['chr']}_{x['start']}_{x['end']}_{x['strand']}", axis=1)

in_bed_df.to_csv(in_bed_with_ids, sep="\t", index=False, header=None)

# bed_overlap_with_genomic_features(in_bed_with_ids, feat_bed, out_file, int_whole_nr=True, use_feat_sc=False)

print("Extracting longest transcripts...")
gtf_extract_most_prominent_transcripts(in_gtf, out_file,
                                           strict=False,
                                           min_len=False,
                                           report=False,
                                           return_ids_dic=tr_ids_dic,
                                           set_ids_dic=False,
                                           add_infos=False)


# Save the tr_ids_dic that is returned by gtf_extract_prominent_transcripts() and reload it instead of recomputing ~ cause it's slow
print("Saving tr_ids_dic")
with open('saved_tr_ids_dic.pkl', 'wb') as f:
    pickle.dump(tr_ids_dic, f)

print("Opening tr_ids_dic")
with open('saved_tr_ids_dic.pkl', 'rb') as f:
    tr_ids_dic = pickle.load(f)

# This depends on the method
SPLIT_SIZE = 81
print("Annotating with exon intron annotations...")
bed_get_exon_intron_annotations_from_gtf(tr_ids_dic, in_bed_with_ids,
                                             in_gtf, eia_out,
                                             stats_dic=None,
                                             own_exon_bed=False,
                                             split_size=SPLIT_SIZE,
                                             n_labels=True,
                                             intron_border_labels=False)