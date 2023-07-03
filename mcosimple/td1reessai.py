import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

#importation du fichier exemple1.xlsx
#definition du variable 
df=pd.read_excel("G:\economie\econometrie\mcosimple\exemple11.xlsx",index_col=0)
df.index=pd.to_datetime(df.index,format='%Y')
print(df)