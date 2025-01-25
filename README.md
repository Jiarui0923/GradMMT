# Explainability for Multi-Modal Transformers with Applications to CD4+ T Cell Epitope Prediction

`Update: 2025-01-24`

CD4+ T cell responses play a crucial role in long-term adaptive immunity. The interaction between T cell receptors and antigen-presenting cells represents the final step in triggering this response and is essential for accurate CD4+ T cell epitope prediction. Recently, multi-modal transformer models like TULIP have effectively incorporated diverse inputs, including CDR sequences, epitope sequences, and MHCII types, to enhance epitope prediction. However, these models suffer from a lack of interpretability, functioning as black boxes. While methods exist to interpret standard transformer models, no tools are currently available for interpreting multi-modal transformers.

In this study, we introduce GradMMT (Gradient-based Rollout for Multi-Modal Transformers), a novel approach to interpret multi-modal transformers. GradMMT extends existing methods by leveraging a unique loss function that enables the separation and interpretation of attention weights across individual encoder blocks. To showcase its utility, we applied GradMMT to the predictions of the TULIP transformer for epitope prediction. We conducted four case studies on T cell responses to three antigens: the SARS-CoV-2 spike protein, the IEa-MHCII molecule, and PE-III from pseudomonas exotoxin. In each case, GradMMT provided insights into the molecular decision-making process of TULIP. This interpretability not only validated TULIP’s predictions but also aligned with experimentally observed findings on the behavior of these antigens in T cell responses.

## Requirements
_TBD_

## Datasets
_TBD_

## Methods
_TBD_

## Citation
_TBD_

## References
The TULIP model code is refactorized from TULIP repository (https://github.com/barthelemymp/TULIP-TCR/)

> Meynard-Piganeau, B., Feinauer, C., Weigt, M., Walczak, A. M., & Mora, T. (2024). TULIP: A transformer-based unsupervised language model for interacting peptides and T cell receptors that generalizes to unseen epitopes. Proceedings of the National Academy of Sciences, 121(24), e2316401121.