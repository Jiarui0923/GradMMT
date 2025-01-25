import pandas as pd

def standarize_dataframe(df, mapping, miss_symbol = '<MIS>'):
    columns = ['CDR3a', 'CDR3b', 'MHC', 'peptide', 'binder']
    std_df = pd.DataFrame([])
    for col in columns:
        if mapping[col] is not None and isinstance(mapping[col], str):
            std_df[col] = df[mapping[col]]
    for col in columns:
        if mapping[col] is None:
            std_df[col] = miss_symbol
    if isinstance(mapping['binder'], (int, float, bool)):
        std_df['binder'] = mapping['binder']
    return std_df