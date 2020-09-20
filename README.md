**DIVERSE: Bayesian Data IntegratiVE learning for
precise drug ResponSE prediction**

This repository contains the implementation of the computational methods in the paper Güvenç Paltun _et al_., DIVERSE: Bayesian Data IntegratiVE learning for
precise drug ResponSE prediction.

Main files:

- `Diverse_out.py` Python script for performing out-of-matrix cross-validation experiments.
- `Diverse_in.py` Python script for performing in-matrix cross-validation experiments.
- `metrics.py`  Contains functions for computing mean square error (MSE) and drug-averaged Sc (Spearman correlation)  evaluation scores

Source files [1]: Folder containing the source code for DIVERSE methdod.

**Requirements**

- Python 2.7
- numpy
- scipy

**Data**

Pre-processed datasets are available in the `DIVERSE/data/original_data` director

- `gdsc_patient_drug_sorted.xlsx` : Drug response data (IC50 values hat gives the effectiveness of drugs on different cell lines) for 956 cancer cell lines and 265 drugs [2].
- `gdsc_patient_gene_sorted.xlsx` : Gene expreesion data  that consists of 232 genes with their interactions with 956 cell lines [2].
- `pubchem_drug_similarity_sorted.xlsx` : Drug similarity data, based on the chemical structural similarity between compounds, is usually used to identify compounds sharing similar biological or chemical activity [3].
- `string_gene_interactions_sorted.csv` : Gene interactions which includes physical and functional associations [4].
- `gdsc_chembl_gene_drug_inteaction_sorted.csv` : Drug–target interaction data which includes interactions for 255 drugs [5].


Processed datasets are available in the `DIVERSE/data `director : after the pre-processing, we obtained 255 drugs, 956 cell lines and 232 genes. For the consistency between integrated data sources, all data sets were scaled to the range between [0,1]. 


**Contact**

Betül Güvenç Paltun, betul.guvenc@aalto.fi

Work was done in the Probabilistic Machine Learning research group at Aalto University.

**Reference**

[1] Brouwer, T. et al (2017) Bayesian Hybrid Matrix Factorisation for Data Integration, Proceedings of Machine Learning Research, 54, 557–566.

[2] Yang, W. et al (2012) Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells, Nucleic acids research, 41,
D955–D961.

[3] Kim, S. et al (2015) PubChem substance and compound databases, Nucleic acids research, 44, D1202–D1213.

[4] Szklarczyk, D. et al (2010) The STRING database in 2011: functional interaction networks of proteins, globally integrated and scored, Nucleic acids research, 39, D561–D568.

[5] Gaulton, A. et al (2017) The ChEMBL database in 2017, Bioinformatics45, D945– D954.

