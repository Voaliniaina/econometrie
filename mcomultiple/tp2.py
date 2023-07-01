import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
# Génération de données aléatoires
np.random.seed(0) # pour la reproductibilité
X1 = np.random.rand(100)
X2 = np.random.rand(100)
beta0 = 1
beta1 = 2
beta2 = 3
epsilon = np.random.randn(100)
Y = beta0 + beta1 * X1 + beta2 * X2 + epsilon
# Création d'un DataFrame
df = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
print(df)
#specification du modele
# Définition des variables dépendantes et indépendantes
X = df[['X1', 'X2']]
y = df['Y']
X = sm.add_constant(X)
model = sm.OLS(y, X)
# Estimation des paramètres du modèle
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

#independance des erreurs(voir la valeur de durbing watson)

print('Statistique de Durbin-Watson:', durbin_watson(results.resid))

#homoscedasticite(optionnel)
plt.scatter(results.fittedvalues, results.resid)
plt.xlabel('Valeurs prédites')
plt.ylabel('Résidus')
plt.title('Résidus vs Valeurs prédites')
plt.show()


# Calculer le test de Breusch-Pagan
bp_test = het_breuschpagan(results.resid, results.model.exog)
# Imprimer les résultats
labels = ['Statistique de test de Lagrange multiplier', 'p-valeur de LM',
'Statistique de test à base de F', 'p-valeur de F']
print(dict(zip(labels, bp_test)))

#normalite des residus

# Calculer la statistique de test de Jarque-Bera et la p-valeur
jb_stats = stats.jarque_bera(results.resid)
jb_stats

#colinearite
# Calcul du VIF
VIF = pd.DataFrame()
VIF["Variable"] = X.columns
VIF["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(VIF)
