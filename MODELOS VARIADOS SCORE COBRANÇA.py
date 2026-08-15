# Bibliotecas padrão
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

#----------------------------------------------
#-- Função para cálculo de KS2
#----------------------------------------------
def KS2(y, y_pred):
    df_ks2 = pd.DataFrame([x for x in y_pred], columns=['REGRESSION_RLog'])
    df_ks2['ALVO'] = [x for x in y]
    return ks_2samp(df_ks2.loc[df_ks2.ALVO==0,"REGRESSION_RLog"], df_ks2.loc[df_ks2.ALVO==1,"REGRESSION_RLog"])[0]
#----------------------------------------------

#--------------------------------------------------------
#-- Funções mostrar fórmula das regressões em SQL Server
#--------------------------------------------------------
def LogisticFormulaSQL(model, X):
    print('Regressão Logística')
    print('SCORE = ROUND(1/(1 + exp(-(       ' + str(model.intercept_[0]))
    for i in range(0, len(model.coef_[0])):
        print('             + {:30}     *     {:.6}'.format(X.columns[i], str(model.coef_[0,i])))
    print('             ))) * 100,2) 	')

def LinearFormulaSQL(model, X):
    print('Regressão Linear')
    print('SCORE = ' + str(model.intercept_))
    for i in range(0, len(model.coef_)):
        print('             + {:30}     *     {:.6}'.format(X.columns[i], str(model.coef_[i])))
#--------------------------------------------------------
        
        
## Carregando os dados

print()
print('----------------     INÍCIO DE CARGA DE DADOS   ------------------------------------')

## Carregando os dados
dataset = pd.read_csv('DATASET_NN.TXT',sep=';') # Separador ;

print('-----------------     FIM DA CARGA DE DADOS   --------------------------------------')
# ---------------------------------------------------------------------------
# Preparando os cojuntos
# ---------------------------------------------------------------------------


cols_in =  [
       'PRE_ATRASO'
      ,'PRE_PROD_CREDICOMP'
      ,'PRE_PROD_COMPJUR'
      ,'PRE_PROD_CARTAO'
      ,'PRE_SEGMENTO_CLASSEA'
      ,'PRE_BOL_VIGENCIA'
      ,'PRE_EXCECAO'
      ,'PRE_SALDO_ALTO'
      ,'PRE_PRIORIDADE'
      ,'PRE_A01'
      ,'PRE_A02'
      ,'PRE_A03'
      ,'PRE_N_CORRENTISTA'
      ,'PRE_VALOR_AVISTA'
      ,'PRE_RENDA'
      ,'PRE_CPC'
      ,'PRE_CORTE_SCORE_FIM'
      ,'PRE_RADAR_INTERNO'
      ,'PRE_REGIAO_SUDESTE'
      ,'PRE_REGIAO_NORDESTE'
      ,'PRE_REGIAO_DEMAIS'
      ,'PRE_CLASSE'
      ]


##------------------------------------------------------------
## Separando em dados de treinamento e teste com Oversampling
##------------------------------------------------------------
y = dataset['PRE_CLASSE']
X = dataset[cols_in]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 122)

# Replica o número de bons - OVersampling
X_train_1 = X_train.query("PRE_CLASSE== 1")
X_train = pd.concat([X_train, X_train_1,X_train_1,X_train_1,X_train_1,X_train_1,X_train_1,X_train_1,X_train_1] , sort = False)

# Refaz o y_train
y_train = X_train['PRE_CLASSE']

#retirar alvo das variáveis de entrada
del X_train['PRE_CLASSE'] 


#---------------------------------------------------------------------------
## Selecionando Atributos
#---------------------------------------------------------------------------
# feature extraction
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression(solver='newton-cg')
selected = RFE(model,15).fit(X_train, y_train)

print()
print('----------------     SELEÇÃO DE VARIÁVEIS--------------------------------------')
print("Num Features: %d" % selected.n_features_)
used_cols = []
for i in range(0, len(selected.support_)):
    if selected.support_[i]: 
        used_cols.append(X_train.columns[i]) 
        print('             -> {:30}     '.format(X_train.columns[i]))
        
print('-------------------------------------------------------------------------------')

X_train = X_train[used_cols]     # Carrega colunas de entrada selecionadas
X_test = X_test[used_cols]       # Carrega colunas de entrada selecionadas

#---------------------------------------------------------------------------
## Ajustando modelos - Aprendizado supervisionado  
#---------------------------------------------------------------------------

# Árvore de decisão com dados de treinamento
from sklearn.tree import DecisionTreeClassifier
#dtree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtree = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=40, min_samples_split=50,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=0, splitter='best')
dtree.fit(X_train, y_train)

# Regressão linear com dados de treinamento
from sklearn.linear_model import LinearRegression
LinearReg = LinearRegression(fit_intercept=True)
LinearReg.fit(X_train, y_train)

# Regressão logística com dados de treinamento
#from sklearn.linear_model import LogisticRegression
LogisticReg = LogisticRegression(solver='newton-cg')
LogisticReg.fit(X_train, y_train)

#Rede Neural com dados de treinamento
from sklearn.neural_network import MLPClassifier 
RNA = MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=True,
       epsilon=1e-08, hidden_layer_sizes=(25), learning_rate='constant',
       learning_rate_init=0.001, max_iter=2000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.3, verbose=False,
       warm_start=False)
RNA.fit(X_train, y_train)


# Bagging com dados de treinamento
from sklearn.ensemble import BaggingClassifier
Bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=25, random_state=10)
Bagging.fit(X_train, y_train)

# RandomForest com dados de treinamento
from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier(n_estimators=20, max_depth=10, min_samples_split=60, random_state=10)
RandomForest.fit(X_train, y_train)

#---------------------------------------------------------------------------
## Previsão treinamento e teste - CLASSIFICAÇÃO
#---------------------------------------------------------------------------
# Árvore de Decisão
y_pred_train_DT = dtree.predict(X_train)
y_pred_test_DT  = dtree.predict(X_test)

# Regressão Linear
y_pred_train_RL = [1 if x > 0.5 else 0 for x in LinearReg.predict(X_train)] 
y_pred_test_RL  = [1 if x > 0.5 else 0 for x in LinearReg.predict(X_test)]

# Regressão Logística
y_pred_train_RLog = LogisticReg.predict(X_train)
y_pred_test_RLog  = LogisticReg.predict(X_test)

# Redes Neurais
y_pred_train_RNA = RNA.predict(X_train)
y_pred_test_RNA = RNA.predict(X_test)
#y_pred_test_RNA  = [1 if x > 0.55 else 0 for x in RNA.predict_proba(X_test)[:,1]] 

# Bagging
y_pred_train_BAG = Bagging.predict(X_train)
y_pred_test_BAG  = Bagging.predict(X_test)

# RandomForest
y_pred_train_RF = RandomForest.predict(X_train)
y_pred_test_RF  = RandomForest.predict(X_test)



#---------------------------------------------------------------------------
## Cálcula e mostra a Acurácia dos modelos
#---------------------------------------------------------------------------
from sklearn import metrics
print()
print('----------     ACURÁCIA     ------------------------------------------------')
print('Acurácia Árvore de Decisão:',metrics.accuracy_score(y_test, y_pred_test_DT))
print('Acurácia Regressão Linear:',metrics.accuracy_score(y_test, y_pred_test_RL))
print('Acurácia Regressão Logística:',metrics.accuracy_score(y_test, y_pred_test_RLog))
print('Acurácia Redes Neurais:',metrics.accuracy_score(y_test, y_pred_test_RNA))
print('Acurácia Bagginh:',metrics.accuracy_score(y_test, y_pred_test_BAG ))
print('Acurácia Randon Forest:' ,metrics.accuracy_score(y_test, y_pred_test_RF))
print('----------------------------------------------------------------------------')


#---------------------------------------------------------------------------
## Previsão treinamento e teste - REGRESSÂO
#---------------------------------------------------------------------------
# Árvore de Decisão
y_pred_train_DT_R  = dtree.predict_proba(X_train)[:,1]
y_pred_test_DT_R  = dtree.predict_proba(X_test)[:,1]

# Regressão Linear
y_pred_train_RL_R = LinearReg.predict(X_train)
y_pred_test_RL_R  = LinearReg.predict(X_test)

# Regressão Logística
y_pred_train_RLog_R = LogisticReg.predict_proba(X_train)[:,1]
y_pred_test_RLog_R  = LogisticReg.predict_proba(X_test)[:,1]

# Redes Neurais
y_pred_train_RNA_R = RNA.predict_proba(X_train)[:,1]
y_pred_test_RNA_R  = RNA.predict_proba(X_test)[:,1]

# Bagging
y_pred_train_BAG_P = Bagging.predict_proba(X_train)
y_pred_test_BAG_P  = Bagging.predict_proba(X_test)

# RandomForest
y_pred_train_RF_P = RandomForest.predict_proba(X_train)
y_pred_test_RF_P  = RandomForest.predict_proba(X_test)

#---------------------------------------------------------------------------
## Cálcula e mostra MSE dos modelos
#---------------------------------------------------------------------------

from math import sqrt
print()
print('----------     RMSE ERROR    ------------------------------------------------')
print('Árvore de Decisão:',    sqrt(np.mean((y_test - y_pred_test_DT_R) **2) ))
print('Regressão Linear:',     sqrt(np.mean((y_pred_test_RL_R -  y_test) ** 2) ))
print('Regressão Logística:',  np.mean((y_pred_test_RLog_R - y_test) ** 2) ** 0.5)
print('Redes Neurais:',        np.mean((y_pred_test_RNA_R - y_test) ** 2) ** 0.5)
print('Bagging:', np.mean((y_pred_test_BAG_P[:,1] - y_test) ** 2))
print('Random Forest:', np.mean((y_pred_test_RF_P[:,1] - y_test) ** 2))
print('----------------------------------------------------------------------------')

#---------------------------------------------------------------------------
## Cálcula o KS2
#---------------------------------------------------------------------------
print()
print('----------------     KS2    ------------------------------------------------')
print('Árvore de Decisão:: ',KS2(y_test,y_pred_test_DT_R))
print('Regressão Linear: ',KS2(y_test,y_pred_test_RL_R))
print('Regressão Logística: ',KS2(y_test,y_pred_test_RLog_R))
print('Redes Neurais: ',KS2(y_test,y_pred_test_RNA_R))
print('Bagging: ',KS2(y_test,y_pred_test_BAG_P[:,1]))
print('Random Forest: ',KS2(y_test,y_pred_test_RF_P[:,1]))
print('----------------------------------------------------------------------------')


#---------------------------------------------------------------------------
## Mostra as fórmulas das regressões para implantação em SQL Server
#---------------------------------------------------------------------------
'''
print()
print('----------------   REPRESENTAÇÃO DOS MODELOS EM SQL SERVER  ----------------')
LogisticFormulaSQL(LogisticReg, X_train)
print()
LinearFormulaSQL(LinearReg, X_train)
print('----------------------------------------------------------------------------')

'''


#----------------------------------------------------------------------
## Montando um Data Frame (Matriz) com os resultados
#----------------------------------------------------------------------
# Conjunto de treinamento
df_train = pd.DataFrame(y_pred_train_DT_R, columns=['REGRESSION_DT'])
df_train['CLASSIF_DT'] = y_pred_train_DT
df_train['REGRESSION_RL'] = y_pred_train_RL_R
df_train['CLASSIF_RL'] =  [1 if x > 0.5 else 0 for x in y_pred_train_RL]
df_train['REGRESSION_RLog'] = y_pred_train_RLog_R
df_train['CLASSIF_RLog'] = y_pred_train_RLog
df_train['REGRESSION_RNA'] = y_pred_train_RNA_R
df_train['CLASSIF_RNA'] = y_pred_train_RNA
df_train['ALVO'] = [x for x in y_train]
df_train['TRN_TST'] = 'TRAIN'
df_train['CLASSIF_BAG'] = y_pred_train_BAG
df_train['REGRESSION_BAG'] = [x for x in y_pred_train_BAG_P[:,1]]
df_train['CLASSIF_RF'] = y_pred_train_RF
df_train['REGRESSION_RF'] = [x for x in y_pred_train_RF_P[:,1]]

# Conjunto de test
df_test = pd.DataFrame(y_pred_test_DT_R, columns=['REGRESSION_DT'])
df_test['CLASSIF_DT'] = y_pred_test_DT
df_test['REGRESSION_RL'] = y_pred_test_RL_R
df_test['CLASSIF_RL'] =  [1 if x > 0.5 else 0 for x in y_pred_test_RL]
df_test['REGRESSION_RLog'] = y_pred_test_RLog_R
df_test['CLASSIF_RLog'] = y_pred_test_RLog
df_test['REGRESSION_RNA'] = y_pred_test_RNA_R
df_test['CLASSIF_RNA'] = y_pred_test_RNA
df_test['ALVO'] = [x for x in y_test]
df_test['TRN_TST'] = 'TEST' 
df_test['CLASSIF_BAG'] = y_pred_test_BAG
df_test['REGRESSION_BAG'] = [x for x in y_pred_test_BAG_P[:,1]]
df_test['CLASSIF_RF'] = y_pred_test_RF
df_test['REGRESSION_RF'] = [x for x in y_pred_test_RF_P[:,1]]


print()
print('----------------    INÍCIO DA EXPORTAÇÃO RESULTADOS   ----------------------------------')
# Juntando Conjunto de Teste e Treinamento
df_total = pd.concat([df_test, df_train], sort = False)

## Exportando os dados para avaliação dos resultados em outra ferramenta
df_test.to_csv('resultado_score_itau_cnc_ativo_V2.csv')

df_total.to_csv('resultado_df_total.csv')
print('----------------     FIM DA EXPORTAÇÃO RESULTADOS   ------------------------------------')

