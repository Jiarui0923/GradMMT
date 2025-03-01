# Explainability for Multi-Modal Transformers with Applications to CD4+ T Cell Epitope Prediction

`Update: 2025-01-24`

CD4+ T cell responses play a crucial role in long-term adaptive immunity. The interaction between T cell receptors and antigen-presenting cells represents the final step in triggering this response and is essential for accurate CD4+ T cell epitope prediction. Recently, multi-modal transformer models like TULIP have effectively incorporated diverse inputs, including CDR sequences, epitope sequences, and MHCII types, to enhance epitope prediction. However, these models suffer from a lack of interpretability, functioning as black boxes. While methods exist to interpret standard transformer models, no tools are currently available for interpreting multi-modal transformers.

In this study, we introduce GradMMT (Gradient-based Rollout for Multi-Modal Transformers), a novel approach to interpret multi-modal transformers. GradMMT extends existing methods by leveraging a unique loss function that enables the separation and interpretation of attention weights across individual encoder blocks. To showcase its utility, we applied GradMMT to the predictions of the TULIP transformer for epitope prediction. We conducted four case studies on T cell responses to three antigens: the SARS-CoV-2 spike protein, the IEa-MHCII molecule, and PE-III from pseudomonas exotoxin. In each case, GradMMT provided insights into the molecular decision-making process of TULIP. This interpretability not only validated TULIP’s predictions but also aligned with experimentally observed findings on the behavior of these antigens in T cell responses.

||CDR3b|Epitope|
|:--|:---:|:------|
|Structure|<img src="imgs/s-cdr3b-r.png" width="200pt">|<img src="imgs/s-epitope-r.png" width="200pt">|
|GradMMT|W95|E24,Q22,K21|

_IEa-MHCII: the core region and secondary region analysis on RCSB 4P46_

> 1. Stadinski, B. D., Trenh, P., Duke, B., Huseby, P. G., Li, G., Stern, L. J., & Huseby, E. S. (2014). Effect of CDR3 sequences and distal V gene residues in regulating TCR–MHC contacts and ligand specificity. The Journal of Immunology, 192(12), 6071-6082.

## Requirements
The dependencies could be installed by `pip install -r requirements.txt`.

The development Python version is `3.8.18` with Ubuntu 20.04.6 LTS (focal). The CUDA version is 12.2.

|Package|Version|Usage|Website|
|:------|:-----:|:----|:-----:|
|matplotlib <img src="https://matplotlib.org/_static/logo_light.svg" width="54pt">|`3.7.3`|Plot figures|[<img src="/imgs/icons/link.png" width="20pt">](https://matplotlib.org/)|
|numpy <img src="https://numpy.org/images/logo.svg" width="16pt">|`1.24.4`|Mathmatical computation|[<img src="/imgs/icons/link.png" width="20pt">](https://numpy.org/)|
|pandas <img src="https://pandas.pydata.org/docs/_static/pandas.svg" width="52pt">|`2.0.3`|Data processing|[<img src="/imgs/icons/link.png" width="20pt">](https://pandas.pydata.org/)|
|scikit-learn <img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width="42pt">|`1.3.2`|Scientific compuation|[<img src="/imgs/icons/link.png" width="20pt">](https://scikit-learn.org/stable/)|
|tqdm <img src="https://tqdm.github.io/img/logo-trans.gif" width="8pt">|`4.66.1`|Progress display|[<img src="/imgs/icons/link.png" width="20pt">](https://tqdm.github.io/docs/tqdm/)|
|torch <img src="https://pytorch.org/assets/images/logo-icon.svg" width="16pt">|`2.2.1`|GPU computation|[<img src="/imgs/icons/link.png" width="20pt">](https://pytorch.org/)|
|transformers 🤗|`4.32.1`|Bert Blocks|[<img src="/imgs/icons/link.png" width="20pt">](https://huggingface.co/docs/transformers/index)|
|tokenizers 🤗|`0.13.3`|Auto Tokenizers|[<img src="/imgs/icons/link.png" width="20pt">](https://huggingface.co/docs/tokenizers/index)|
|tabulate |`0.9.0`|Table Format|[<img src="/imgs/icons/link.png" width="20pt">](https://pypi.org/project/tabulate/)|
|docflow |`git:1.0.0`|Report Generation|[<img src="/imgs/icons/link.png" width="20pt">](https://github.com/Jiarui0923/DocFlow)|

**NOTICE**: `DocFlow` should be installed from GitHub repository.

## Datasets
The datasets utilized in the paper.

### SARS-CoV-2 Spike
- Dataset Path: [`datasets/sars-cov-2/sars-cov-2-structure-vdjdb.csv`](datasets/sars-cov-2/sars-cov-2-structure-vdjdb.csv)
- 3 samples
- CDRa: YES
- CDRb: YES
- Peptide: YES
- MHC: YES

> 1. Mudd, P. A., Minervina, A. A., Pogorelyy, M. V., Turner, J. S., Kim, W., Kalaidina, E., ... & Ellebedy, A. H. (2022). SARS-CoV-2 mRNA vaccination elicits a robust and persistent T follicular helper cell response in humans. Cell, 185(4), 603-613.
> 2. Rowntree, L. C., Nguyen, T. H., Kedzierski, L., Neeland, M. R., Petersen, J., Crawford, J. C., ... & Kedzierska, K. (2022). SARS-CoV-2-specific T cell memory with common TCRαβ motifs is established in unvaccinated children who seroconvert after infection. Immunity, 55(7), 1299-1315.
> 3. Bagaev, D. V., Vroomans, R. M., Samir, J., Stervbo, U., Rius, C., Dolton, G., ... & Shugay, M. (2020). VDJdb in 2019: database extension, new analysis infrastructure and a T-cell receptor motif compendium. Nucleic acids research, 48(D1), D1057-D1062.

### IEa-MHCII Self-antigen

The structure study RCSB Crystallography Structure: RCSB 4P46

- Dataset Path: [`datasets/iea-mhcii/iea-mhcii-structure.csv`](datasets/iea-mhcii/iea-mhcii-structure.csv)
- 10 samples
- CDRa: YES
- CDRb: YES
- Peptide: YES
- MHC: NO

> 1. Stadinski, B. D., Trenh, P., Duke, B., Huseby, P. G., Li, G., Stern, L. J., & Huseby, E. S. (2014). Effect of CDR3 sequences and distal V gene residues in regulating TCR–MHC contacts and ligand specificity. The Journal of Immunology, 192(12), 6071-6082.

### PE-III from pseudomonas exotoxin
_TBD_

## Methods
The steps to run GradMMT on your data or reproduce the results in paper.
### 1. Build Dataset
_If you are reproducing the results, please skip this step._  
Please use function `gradmmt.standarize_dataframe` to transform your dataset to GradMMT readable format. The guildeline could be found at [`dataset.ipynb`](dataset.ipynb).
### 2. One-click to Run GradMMT
`gradmmt.RolloutExperiment` enables to run GradMMT with TULIP with just one-click. Please make sure the inputed data has already been standarized by the step 1.
The guildeline could be found at [`gradmmt.ipynb`](gradmmt.ipynb).

## Citation
_TBD_

## References
The TULIP model code is refactorized from TULIP repository (https://github.com/barthelemymp/TULIP-TCR/)

> 1. Meynard-Piganeau, B., Feinauer, C., Weigt, M., Walczak, A. M., & Mora, T. (2024). TULIP: A transformer-based unsupervised language model for interacting peptides and T cell receptors that generalizes to unseen epitopes. Proceedings of the National Academy of Sciences, 121(24), e2316401121.