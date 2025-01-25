# Explainability for Multi-Modal Transformers with Applications to CD4+ T Cell Epitope Prediction

`Update: 2025-01-24`

CD4+ T cell responses play a crucial role in long-term adaptive immunity. The interaction between T cell receptors and antigen-presenting cells represents the final step in triggering this response and is essential for accurate CD4+ T cell epitope prediction. Recently, multi-modal transformer models like TULIP have effectively incorporated diverse inputs, including CDR sequences, epitope sequences, and MHCII types, to enhance epitope prediction. However, these models suffer from a lack of interpretability, functioning as black boxes. While methods exist to interpret standard transformer models, no tools are currently available for interpreting multi-modal transformers.

In this study, we introduce GradMMT (Gradient-based Rollout for Multi-Modal Transformers), a novel approach to interpret multi-modal transformers. GradMMT extends existing methods by leveraging a unique loss function that enables the separation and interpretation of attention weights across individual encoder blocks. To showcase its utility, we applied GradMMT to the predictions of the TULIP transformer for epitope prediction. We conducted four case studies on T cell responses to three antigens: the SARS-CoV-2 spike protein, the IEa-MHCII molecule, and PE-III from pseudomonas exotoxin. In each case, GradMMT provided insights into the molecular decision-making process of TULIP. This interpretability not only validated TULIP’s predictions but also aligned with experimentally observed findings on the behavior of these antigens in T cell responses.

## Requirements
The dependencies could be installed by `pip install -r requirements.txt`.

The development Python version is `3.8.18` with Ubuntu 20.04.6 LTS (focal). The CUDA version is 12.2.

|Package|Version|Usage|Website|
|:------|:-----:|:----|:-----:|
|matplotlib <img src="https://matplotlib.org/_static/logo_light.svg" width="54pt">|`3.7.3`|Plot figures|[<img src="/imgs/icons/link.png" width="20pt">](https://matplotlib.org/)|
|numpy <img src="https://numpy.org/images/logo.svg" width="16pt">|`1.24.4`|Mathmatical computation|[<img src="/imgs/icons/link.png" width="20pt">](https://numpy.org/)|
|pandas <img src="https://pandas.pydata.org/docs/_static/pandas.svg" width="52pt">|`2.0.3`|Data processing|[<img src="/imgs/icons/link.png" width="20pt">](https://pandas.pydata.org/)|
|scikit-learn <img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width="42pt">|`1.3.2`|Scientific compuation|[<img src="/imgs/icons/link.png" width="20pt">](https://scikit-learn.org/stable/)|
|tqdm <img src="https://tqdm.github.io/img/logo-trans.gif" width="16pt">|`4.66.1`|Progress display|[<img src="/imgs/icons/link.png" width="20pt">](https://tqdm.github.io/docs/tqdm/)|
|torch <img src="https://pytorch.org/assets/images/logo-icon.svg" width="16pt">|`2.2.1`|GPU computation|[<img src="/imgs/icons/link.png" width="20pt">](https://pytorch.org/)|
|transformers 🤗|`4.32.1`|Bert Blocks|[<img src="/imgs/icons/link.png" width="20pt">](https://huggingface.co/docs/transformers/index)|
|tokenizers 🤗|`0.13.3`|Auto Tokenizers|[<img src="/imgs/icons/link.png" width="20pt">](https://huggingface.co/docs/tokenizers/index)|
|tabulate |`0.9.0`|Table Format|[<img src="/imgs/icons/link.png" width="20pt">](https://pypi.org/project/tabulate/)|
|docflow |`git:1.0.0`|Report Generation|[<img src="/imgs/icons/link.png" width="20pt">](https://github.com/Jiarui0923/DocFlow)|

**NOTICE**: `DocFlow` should be installed from GitHub repository.

## Datasets
_TBD_

## Methods
_TBD_

## Citation
_TBD_

## References
The TULIP model code is refactorized from TULIP repository (https://github.com/barthelemymp/TULIP-TCR/)

> Meynard-Piganeau, B., Feinauer, C., Weigt, M., Walczak, A. M., & Mora, T. (2024). TULIP: A transformer-based unsupervised language model for interacting peptides and T cell receptors that generalizes to unseen epitopes. Proceedings of the National Academy of Sciences, 121(24), e2316401121.