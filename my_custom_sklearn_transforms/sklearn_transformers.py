from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

# All sklearn Transforms must have the `transform` and `fit` methods
class BalanceData(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        #se separan los datos de la variable objetivo que estan desbalanceados
        df_majority = data[data.OBJETIVO == "Aceptado"]
        df_minority = data[data.OBJETIVO == "Sospechoso"]

        #se igualan ejemplos para el balanceo
        df_minority_upsampled = resample(df_minority,
                                         replace=True,
                                         n_samples=8873,
                                         random_state=0)
        #se retorna la data balanceada
        return pd.concat([df_majority, df_minority_upsampled])