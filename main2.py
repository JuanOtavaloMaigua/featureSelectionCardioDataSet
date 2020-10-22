import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import scipy.stats as stats
from scipy.stats import chi2_contingency
import random as rd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

columnaNombres = ['F1R', 'F1S', 'F2R', 'F2S', 'F3R', 'F3S', 'F4R', 'F4S', 'F5R', 'F5S', 'F6R', 'F6S', 'F7R', 'F7S',
                  'F8R', 'F8S', 'F9R', 'F9S', 'F10R', 'F10S', 'F11R', 'F11S', 'F12R', 'F12S', 'F13R', 'F13S', 'F14R',
                  'F14S', 'F15R', 'F15S', 'F16R', 'F16S', 'F17R', 'F17S', 'F18R', 'F18S', 'F19R', 'F19S', 'F20R',
                  'F20S', 'F21R', 'F21S', 'F22R', 'F22S']


class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None  # P-Value
        self.chi2 = None  # Chi Test Statistic
        self.dof = None

        self.dfObserved = None
        self.dfExpected = None


    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p < alpha:
            result = "{0} is IMPORTANT for Prediction".format(colX)
        else:
            result = "{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        return result

    def TestIndependence(self, colX, colY, alpha=0.30):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)

        self.dfObserved = pd.crosstab(Y, X)
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index=self.dfObserved.index)

        return self._print_chisquare_result(colX, alpha)


def readJoinFile(nombreLista):
    listaDataFrames = []
    for nombre in nombreLista:
        matrizDatoExtraido = pd.read_csv(nombre)
        listaDataFrames.append(matrizDatoExtraido)
    dfUnion = pd.concat(listaDataFrames, ignore_index=True)
    target = dfUnion.iloc[:, 0]
    numeroColumnas = len(dfUnion.columns)
    matriz = dfUnion.iloc[:, 1:numeroColumnas]
    return target, matriz

def minMaxNormalizacion(ingresarMatriz):
    #La matriz ingresada debe ser numpy
    # Primera Normalizacion seleccionando valores minimos y maximos de cada columna(vector)
    indexNumbers = list(range(len(ingresarMatriz)))

    scaler = MinMaxScaler()
    scaler.fit(ingresarMatriz)
    minMaxVector = scaler.transform(ingresarMatriz)

    df1 = pd.DataFrame(data = minMaxVector, index= indexNumbers, columns= columnaNombres)
    minMaxVector = df1


    # Segundo Normalizacion seleccionando valores minimos y maximos globales de la matriz
    maximoGeneral = np.amax(ingresarMatriz)
    minimoGeneral = np.amin(ingresarMatriz)

    df2 = pd.DataFrame(data = ingresarMatriz, index= indexNumbers, columns= columnaNombres)

    for nombresColumnas in columnaNombres:
        df2[nombresColumnas] = (df2[nombresColumnas]-minimoGeneral)/(maximoGeneral-minimoGeneral)

    minMaxMatriz = df2

    return minMaxVector, minMaxMatriz

def correlacionPearson(dfCorrelacionado, nombreColumna):
    copyDfCorrelacionado = dfCorrelacionado
    tempNumpy = dfCorrelacionado.to_numpy()
    tempNumpy = np.triu(tempNumpy, k=1)
    dfCorrelacionado = pd.DataFrame(tempNumpy, columns=nombreColumna)

    listaFinalDefinitiva = []
    for nombreVariable in nombreColumna:
        listTemp = (dfCorrelacionado[nombreVariable]).tolist()
        for indice in range(0, len(listTemp)):
            if listTemp[indice]>0.85:
                    listaFinalDefinitiva.append((nombreVariable, nombreColumna[indice]))
    # listaExcluidos = []
    if len(listaFinalDefinitiva)>0:
        for setValores in listaFinalDefinitiva:
            # listaExcluidos.append(list(setValores)[0])
            del copyDfCorrelacionado[list(setValores)[0]]

    # del copyDfCorrelacionado[listaExcluidos[0]]
    return copyDfCorrelacionado

if __name__ == '__main__':
    nombreLista = ['SPECTF.train.csv', 'SPECTF.csv']
    getTarget, getMatriz = readJoinFile(nombreLista)
    print('Matriz no normalizada y sin el target: ')
    print(getMatriz)
    print('------------------------------------')
    print('')

    minMaxVector, minMaxMatriz = minMaxNormalizacion(getMatriz.to_numpy())
    print('Normalizacion por Vector: ')
    print(minMaxVector)
    print(' ')
    print('Normalizacion por Matriz: ')
    print(minMaxMatriz)
    print('-----------------------------------')
    print('')


    cT2 = ChiSquare(minMaxMatriz)
    minMaxMatriz['target'] = getTarget
    listaVariablesIndependientes = []
    for var in columnaNombres:
        variables = cT2.TestIndependence(colX=var, colY='target')
        print(variables)
        if variables.__contains__('is IMPORTANT for Prediction'):
            listaVariablesIndependientes.append(variables.split(' ')[0])

    print('')
    print('-----------------------------------')
    print('')
    print('Lista de Features seleccionados con alfa: 0.30')
    print(listaVariablesIndependientes)
    dfListaVariablesIndependientes = minMaxMatriz[listaVariablesIndependientes]
    print(dfListaVariablesIndependientes)

    print('')
    print('-----------------------------------')
    print('')

    print('Resultado de analisis de correlacion: ')
    dfCorrelacion = dfListaVariablesIndependientes.corr(method='pearson')
    print(dfCorrelacion)

    print('')
    print('-----------------------------------')
    print('')

    print('Valores mayores a 0,85 estan altamente correlacionados')
    analisisCorrelacion = correlacionPearson(dfCorrelacion, listaVariablesIndependientes)
    print(analisisCorrelacion)





# def filterChiSquared()
    # seleccionDeFeatures = SelectKBest(chi2, k=2).fit_transform(X, y)
    # print(seleccionDeFeatures)
    # pipe = Pipeline([
    #     # the reduce_dim stage is populated by the param_grid
    #     ('reduce_dim', 'passthrough'),
    #     ('classify', LinearSVC(dual=False, max_iter=10000))
    # ])
    #
    # N_FEATURES_OPTIONS = [3, 5, 7, 9]
    # C_OPTIONS = [1, 10, 100, 1000]
    # param_grid = [
    #     {
    #         'reduce_dim': [SelectKBest(chi2)],
    #         'reduce_dim__k': N_FEATURES_OPTIONS,
    #         'classify__C': C_OPTIONS
    #     },
    # ]
    #
    # grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid)
    # grid.fit(X, y)
    # mean_scores = np.array(grid.cv_results_['mean_test_score'])
    # print(mean_scores)
    #
    # X_new = SelectKBest(chi2, k=16).fit_transform(X, y)
    #
    # pd.crosstab(np.squeeze(X_new), np.squeeze(y))




# def filterChiCuadrado(X, y):
#     fila = X.shape[0]
#     columna = X.shape[1]
#
#     sumaColumnas = []
#     sumaFilas = []
#
#     for nombreColumna1 in columnaNombres:
#         dfVectorColumna1 = X[nombreColumna1].sum()
#         sumaColumnas.append(dfVectorColumna1)
#
#     for nombreColumna1 in range(0, columna):
#         # print(X.iloc[:,[nombreColumna1]])
#         if nombreColumna1 < columna:
#             for nombreColumna2 in range(nombreColumna1+1,columna):
#                 print('Comparacion entre columna: ',nombreColumna1, nombreColumna2)
#                 sumaInteraccionFilas = (X[columnaNombres.__getitem__(nombreColumna1)] + X[columnaNombres.__getitem__(nombreColumna2)])
#                 sumaTotal = sumaInteraccionFilas.sum()
#                 ((sumaInteraccionFilas*sumaColumnas.__getitem__(nombreColumna1))/sumaTotal)
#
#         print('----------------------------------------------')
#         # for nombreColumna2 in columnaNombres:
#         #     if nombreColumna1 != nombreColumna2:
#         #         dfSumaFilas = X[nombreColumna1] + X[nombreColumna2]
#         #         dfVectorColumna2 = X[nombreColumna2].sum()
#         #         sumaTotal = dfSumaFilas.sum()
#         #         print(dfVectorColumna1, dfVectorColumna2)
#
#                 # print(df)


    # cT = ChiSquare(minMaxVector)
    #
    # minMaxVector['target'] = getTarget
    # for var in columnaNombres:
    #     cT.TestIndependence(colX=var, colY='target')
    #
    # print('---------------------------------------')
    # print('')