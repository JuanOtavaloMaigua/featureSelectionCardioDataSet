import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier

pd.set_option('display.max_row', None)
pd.set_option('display.max_column', None)
# pd.set_option('display.width', 100)

def lecturaDataSet(NombreArchivo):
    archivoDataSetLeido = pd.read_csv(NombreArchivo)
    listaDatosFlotantes = list(map(lambda x: list(map(float, (x[1]).tolist().pop(0).split(';'))),archivoDataSetLeido.iterrows()))
    listaDatos = []
    labelDatos = []
    for lista in listaDatosFlotantes:
        labelDatos.append(lista.pop(0))
        listaDatos.append(lista)

    #Se normalizan los datos
    datosNumpy = np.asarray(listaDatos).transpose()
    scaler = MinMaxScaler()
    scaler.fit(datosNumpy)
    minMaxListaDatos = scaler.transform(datosNumpy)
    listaDatos = minMaxListaDatos

    return listaDatos, labelDatos

def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func= f_classif, k = 3)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


if __name__ == '__main__':
    print('Filtering with Anova')
    # X, y = lecturaDataSet('dataset.csv')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    #
    # model = ExtraTreesClassifier(n_estimators=10)
    # model.fit(X_train, np.ravel(y_train))
    #
    # print(model.feature_importances_)

    df, targetColumn = lecturaDataSet('dataset.csv')

    X_train, X_test, y_train, y_test = train_test_split(df.transpose(), targetColumn, test_size=0.33, random_state=1)

    diccionario = {}
    cont = 0
    for data in df:
        diccionario[str(cont)] = list(data)
        cont = cont + 1

    dfPruebas = pd.DataFrame(diccionario)

    knn = KNeighborsClassifier(n_neighbors=4)
    sfs = SFS(knn,
              k_features=20,
              forward=True,
              floating=False,
              verbose=2,
              scoring='accuracy',
              cv=0)

    knn.fit(X_train, y_train)
    sfs.fit(X_train, y_train)

    conjuntoDeHipotesis = []
    for data in sfs.k_feature_names_.__iter__():
        conjuntoDeHipotesis.append(data)
    print('')
    print(conjuntoDeHipotesis)
    sample = list(map(int, conjuntoDeHipotesis))
    dfPruebas = dfPruebas[conjuntoDeHipotesis]
    score1 = knn.score(dfPruebas.to_numpy(), y_train.to_numpy())
    score2 = knn.score(X_test, y_test)
    print(dfPruebas)
    print(score1,score2)



    # Primer intento con acuracy de 80.90 hay que ver con cuantos features
    # X, y = lecturaDataSet('dataset.csv')
    #
    # # De el conjunto de datos X se toma el 33% como data test, Todo cambiar si es necesario
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    #
    # X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
    # # fit the model
    # model = LogisticRegression(solver='liblinear')
    # model.fit(X_train_fs, y_train)
    # # evaluate the model
    # yhat = model.predict(X_test_fs)
    # # evaluate predictions
    # accuracy = accuracy_score(y_test, yhat)
    # print('Feature list with scores: ')
    # for i in range(len(fs.scores_)):
    #     print('Feature %d: %f' % (i, fs.scores_[i]))
    # print('Accuracy: %.2f' % (accuracy * 100))

    #Segundo intento 0.814 de score son con 10 features y 80,90 con 7 features

    #
    # X, y = lecturaDataSet('dataset.csv')
    # # define the evaluation method
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # # define the pipeline to evaluate
    # model = LogisticRegression(solver='sag')
    # fs = SelectKBest(score_func=f_classif)
    # pipeline = Pipeline(steps=[('anova', fs), ('lr', model)])
    # # define the grid
    # grid = dict()
    # grid['anova__k'] = [i + 1 for i in range(X.shape[1])]
    # # define the grid search
    # search = GridSearchCV(pipeline, grid, scoring='accuracy', n_jobs=-1, cv=cv)
    # # perform the search
    # results = search.fit(X, y)
    # # summarize best
    # print('Best Mean Accuracy: %.3f' % results.best_score_)
    # print('Best Config: %s' % results.best_params_)












