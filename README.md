
````markdown
# Barley-Wheat Salt Stress Candidate Gene Analysis

This repository contains processed data, analysis scripts, model outputs, and final figures for the undergraduate thesis:

**Prediction and Functional Analysis of Key Salt-Tolerance Genes in Barley and Wheat Based on Transcriptomics and MLP Model**

## Project Overview

Soil salinization is a major abiotic stress that restricts crop growth and yield. This project integrates barley and wheat salt-stress transcriptome datasets and applies a Masked MLP model, SHAP interpretation, multi-model voting, comparative genomics, Ka/Ks analysis, promoter cis-element analysis, and qRT-PCR validation to identify candidate salt-responsive genes in barley and wheat.

## Repository Structure

```text
data/       Processed data files used in the thesis
scripts/    Main analysis and plotting scripts
results/    Model training outputs and intermediate results
figures/    Final figures used in the thesis
docs/       Supplementary notes and file descriptions
````

## Data Organization

The `data/` directory is organized as follows:

```text
data/
├── metadata/             RNA-seq dataset metadata
├── expression_matrix/    Processed barley and wheat expression matrices
├── feature_genes/        Top 2000 highly variable gene lists
├── orthology/            Barley-wheat orthology mapping files
├── annotation/           Functional annotation and HMMER annotation files
├── evolution/            Ka/Ks analysis results
├── cis_elements/         PlantCARE cis-element prediction results
├── motif/                Motif and phylogenetic analysis input files
└── qpcr/                 qRT-PCR primer and expression data
```

## Figures

Final thesis figures are stored in the `figures/` directory and organized according to the figure numbers in the thesis:

* `Fig_3_1_Masked_MLP_workflow/`: overview of the Masked MLP-based candidate gene identification workflow
* `Fig_3_2_batch_correction/`: PCA and expression distribution before and after batch correction
* `Fig_3_3_model_performance/`: training dynamics, cross-validation performance, and ROC curves
* `Fig_3_4_SHAP_consensus/`: SHAP interpretation and multi-model voting results
* `Fig_3_5_synteny/`: barley-wheat synteny and orthologous mapping
* `Fig_3_6_KaKs/`: Ka/Ks selection pressure analysis
* `Fig_3_7_phylogeny_motif_structure/`: phylogenetic tree, conserved motifs, and gene structure
* `Fig_3_8_cis_elements/`: promoter cis-acting element distribution and regulatory network
* `Fig_3_9_RNAseq_expression_heatmap/`: RNA-seq expression heatmap under salt stress
* `Fig_3_10_qPCR/`: qRT-PCR validation results

## Main Candidate Genes

Barley candidate genes:

* `HvXB3`
* `HvSTI1`
* `HvUbiA`
* `HvCaM1`

Wheat candidate genes:

* `TaSTI1-B`
* `TaDNAJ-A`
* `TaDDX-A`
* `TaVPS9-B`
* `TaSTK-B`
* `TaRPL32-D`

## Main Workflow

The analysis workflow includes:

1. RNA-seq data collection and metadata curation
2. Expression matrix preprocessing and batch correction
3. Top 2000 highly variable gene selection
4. Masked MLP model training and evaluation
5. SHAP-based feature interpretation and multi-model voting
6. Orthology mapping and synteny analysis between barley and wheat
7. Ka/Ks selection pressure analysis
8. Conserved motif, phylogenetic, and gene structure analysis
9. Promoter cis-acting element analysis
10. RNA-seq expression profiling and qRT-PCR validation

## Data Availability

Processed data files used in the thesis are provided in the `data/` directory. Large raw sequencing files, raw genome files, and reference CDS files are not included in this repository. These files can be downloaded from public databases according to the accession numbers and database sources described in the thesis.

## Code Availability

The main scripts are provided in the `scripts/` directory. The scripts are organized according to the major analysis steps, including batch correction, model training, SHAP interpretation, synteny visualization, Ka/Ks analysis, cis-element analysis, motif visualization, and qRT-PCR plotting.

## Notice

This repository is made publicly available for academic review, thesis evaluation, and reproducibility checking.

Unless otherwise stated, no open-source license is granted for the code, data, figures, or documents in this repository. Redistribution, modification, reuse, or commercial use is not permitted without explicit written permission from the author.

Copyright © 2026 Wan Boyan. All rights reserved.

```
```
