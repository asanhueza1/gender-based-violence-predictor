import pandas as pd
rel_pareja_1 = pd.read_csv("rel_pareja_1.csv", dtype='string')
str_cols = ['ID_VIV', 'ID_MUJ', 'UPM', 'DOMINIO', 'NOM_ENT', 'NOM_MUN', 'T_INSTRUM']
num_cols = rel_pareja_1.columns.to_list()
for col in str_cols:
    num_cols.remove(col)
rel_pareja_1[num_cols] = rel_pareja_1[num_cols].apply(pd.to_numeric, errors='coerce')