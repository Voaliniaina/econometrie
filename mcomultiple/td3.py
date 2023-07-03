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

#créer la colonne de retard
df['log_recettes_lag1']=df['log_recettes'].shift(1)
#eliminer la premiere ligne qui aura une valeur Nan pour le retard
df=df.dropna()
print(df)

#specification du modele
X = df[['log_pib','log_recettes_lag1']]
y = df['log_recettes']
# Ajout d'une constante à nos variables explicatives pib=x recettes=y
X=sm.add_constant(X)
# Spécification du modèle MCO
model=sm.OLS(y,X)
# Estimation des paramètres
results=model.fit()
# Affichage des résultats
print(results.summary())

#h1 linearite
plt.scatter(results.fittedvalues, y)
plt.xlabel('Valeurs prédites')
plt.ylabel('Valeurs observées')
plt.title('Valeurs prédites vs Valeurs observées')
plt.show()