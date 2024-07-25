import pandas as pd
import numpy as np
import os

def dataframe_info(df):
    """
    Retourne une description claire du DataFrame pandas mis en paramètre permettant une analyse plus claire que .info()
    """
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


def racine_projet():
    """
    Retourne le chemin absolu de la racine du projet en remontant de deux niveaux.
    """
    # Obtenir le chemin absolu du répertoire où se trouve ce fichier
    dossier_courant = os.path.dirname(os.path.abspath(__file__))
    # Remonter de deux niveaux pour atteindre la racine du projet
    racine = os.path.dirname(dossier_courant)
    return racine