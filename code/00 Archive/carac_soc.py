carac_soc = pd.read_csv('carac_soc.csv', dtype={'ID_MUJ': 'string', 'NOM_ENT':'string', 'NOM_MUN': 'string', 
                                                'COD_RES_E': 'string', 'NOMBRE': 'string', 'NIV': 'string', 
                                                'GRA': 'string', 'P2_8': 'string', 'P2_9': 'string', 'P2_10': 'string', 
                                                'P2_11': 'string', 'P2_12': 'string', 'P2_13': 'string', 'P2_14': 'string', 
                                                'P2_15': 'string', 'P2_16': 'string', 'CODIGO': 'string', 
                                                'REN_MUJ_EL': 'string', 'REN_INF_AD': 'string', 'FN_DIA': 'string', 
                                                'FN_MES': 'string', 'DOMINIO': 'string', 'COD_M15': 'string'})

sel_lst = [f'P2_{i}' for i in range(5, 17) if i != 7]
sel_lst.extend(['SEXO', 'EDAD', 'NIV', 'GRA'])
carac_soc[sel_lst] = carac_soc[sel_lst].apply(pd.to_numeric, errors='coerce')
