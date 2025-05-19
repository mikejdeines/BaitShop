# BaitShop

Creates a library of encoding probes suitable for MERFISH experiments.\
Required inputs:
* Codebook with gene names in the 'name' column, ensembl transcript ids in the 'id' column, and 1 or 0 values for each readout probe.
* Readout probe sequences (see Xia, C. et al. Proceedings of the National Academy of Sciences 116, 19490–19499 (2019)).
* Forward and reverse primers.
* a cDNA fasta file (e.g. https://ftp.ensembl.org/pub/release-114/fasta/mus_musculus/cdna/Mus_musculus.GRCm39.cdna.all.fa.gz)
* a ncRNA fasta file (e.g. https://ftp.ensembl.org/pub/release-114/fasta/mus_musculus/ncrna/Mus_musculus.GRCm39.ncrna.fa.gz)

Exports a fasta file containing the full probe sequences.

To install:

```
conda create -n MERFISH_probe_design python=3.10
git clone https://github.com/mikejdeines/BaitShop
cd BaitShop
pip install .
```
