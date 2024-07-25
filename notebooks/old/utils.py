import pandas as pd
import numpy as np

def dataframe_info(df):
    # Initialiser une liste pour stocker les informations
    info_list = []

    # Parcourir chaque colonne pour recueillir les informations
    for col in df.columns:
        non_null_count = df[col].notnull().sum()
        dtype = df[col].dtype
        nan_count = df[col].isna().sum()
        nan_percentage = (nan_count / len(df)) * 100
        example_value = df[col].dropna().iloc[0] if non_null_count > 0 else None
        info_list.append([col, non_null_count, nan_count, round(nan_percentage, 2), dtype, example_value])

    # Créer un DataFrame à partir de la liste d'informations
    info_df = pd.DataFrame(info_list, columns=['Column', 'Non-Null Count', 'NaN Count', 'NaN Percentage', 'Dtype', 'Example Value'])
    
    return info_df