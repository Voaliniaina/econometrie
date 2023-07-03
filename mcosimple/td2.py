import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

#importation du fichier exemple1.xlsx
#definition du variable 
df=pd.read_excel("G:\economie\econometrie\mcosimple\exemple11.xlsx",index_col=0)
df.index=pd.to_datetime(df.index,format='%Y')
#transformer les données en un logarithme néperien
df['log_pib']=np.log(df['pib'])
df['log_recettes']=np.log(df['recettes'])
print(df['log_pib'])
print (df['log_recettes'])

#specification du modele
X=df["log_pib"]
y=df["log_recettes"]
# Ajout d'une constante à nos variables explicatives pib=x recettes=y
X=sm.add_constant(X)
# Spécification du modèle MCO
model=sm.OLS(y,X)
# Estimation des paramètres
results=model.fit()
# Affichage des résultats
print(results.summary())
#validation des hypotheses

#h1 linearite
plt.scatter(results.fittedvalues, y)
plt.xlabel('Valeurs prédites')
plt.ylabel('Valeurs observées')
plt.title('Valeurs prédites vs Valeurs observées')
plt.show()