#!/usr/bin/env python
# coding: utf-8

# In[81]:


mujeres_data_13_2 = pd.read_csv("conjunto_de_datos_tb_sec_xiii_2_endireh_2016/conjunto_de_datos/conjunto_de_datos_tb_sec_xiii_2_endireh_2016.csv", dtype = 'string')

str_cols = ['ID_VIV','ID_MUJ','DOMINIO', 'NOM_ENT', 'NOM_MUN', 'T_INSTRUM']
num_cols = mujeres_data_13_2.columns.to_list()
for col in str_cols:
    num_cols.remove(col)

mujeres_data_13_2[num_cols] = mujeres_data_13_2[num_cols].apply(pd.to_numeric, errors='coerce')

