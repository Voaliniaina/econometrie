import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

#importation du fichier exemple1.xlsx
#definition du variable 
df=pd.read_excel("G:\economie\econometrie\mcosimple\exemple11.xlsx",index_col=0)
df.index=pd.to_datetime(df.index,format='%Y')
print(df)

#specification du modele
X=df["pib"]
y=df["recettes"]

# Ajout d'une constante à nos variables explicatives
X = sm.add_constant(X)
# Spécification du modèle MCO
model = sm.OLS(y, X)
# Estimation des paramètres
results = model.fit()
# Affichage des résultats
print(results.summary())

#validation des hypotheses

#h1 linearite
plt.scatter(results.fittedvalues, y)
plt.xlabel('Valeurs prédites')
plt.ylabel('Valeurs observées')
plt.title('Valeurs prédites vs Valeurs observées')
plt.show()

#appliquer une transformation logarithmique
transformed_X=np.log(X)
transformed_y=np.log(y)
print(transformed_X)
print(transformed_X)

#essai transfo log x
transformed_X=np.log(X)
print(transformed_X)

#essai transfo log y
transformed_y=np.log(y)
print(transformed_y)

#essai mr NAPEKARINA ARY AM PD READ EXCEL

