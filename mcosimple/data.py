import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt



#collecte des données
np.random.seed(0)
x=np.linspace(0,100,100)
y=3*x+np.random.normal(0,10,100)
data=pd.DataFrame(data={'x':x,'y':y})
print(data.head())
#specification du modèle
X=data['x']
y=data['y']
#ajout d'une constante
X=sm.add_constant(X)
#specification du modele
model=sm.OLS(y,X)
#estimation des parametres
results=model.fit()
#affichage
print(results.summary())
#test de linearite
predictions = results.predict(X) # calcule des prédictions
residuals = results.resid # calcul des résidus
plt.scatter(predictions, residuals)
plt.axhline(0, color='red')
plt.xlabel('Valeurs prédites')
plt.ylabel('Résidus')
plt.title('Résidus vs Valeurs prédites')
plt.show()