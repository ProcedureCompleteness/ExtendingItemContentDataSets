# parts of this script are based on the paper by Jurek, Hong, Chi, Liu (2017) "A novel ensemble learning approach to unsupervised record linkage"
# (https://www.sciencedirect.com/science/article/abs/pii/S0306437916305063)
# and the github project https://github.com/Mogady/Novel-Ensample-approch-Record-linkage

import pandas as pd
import numpy as np
from itertools import combinations
import itertools
from frameworks.SelfLearning import SelfLearningModel
from sklearn.linear_model import LogisticRegression
from collections import defaultdict

## Step 1 Duplicate Detection
## Task 1.2 item pair classification

# specify number of (non-)duplicate pairs for initialization of self learning
num_duplicates_init_self_learning = 100
num_non_duplicates_init_self_learning = 100

#### Jurek_Step 1 import a blocked pairs dataframe with different similarity metrics for each given feautre
pairs_file_name='pairs.pkl'
pairs= pd.read_pickle(pairs_file_name)
pairs = pairs.fillna(0)
pairs = pairs.sample(frac=1)

#### Jurek_Step 2 generate the feature schemas (similarity schemas)
def cos_sim(v1,v2):
    return np.dot(v1,v2)/np.sqrt(np.dot(v1,v1)*np.dot(v2,v2))
'''
inputs :
    df : the Dateframe containing the features
    cols : list of cols to calculate cosine similarity 
'''
def get_lowest(df,cols):
    low=2
    m1=0
    m2=0
    for k in combinations(cols,2):
        cos=cos_sim(df[k[0]].values,df[k[1]].values)
        if cos<low:
            low=cos
            m1=k[0]
            m2=k[1]
    return m1,m2
'''
inputs :
    df : the Dateframe containing multiple similarities for each feature
    p : cosine similarity threshold
outputs :
    selected features
'''
def generate_pool(df,p):
    features=defaultdict(list)
    for feature in list(set([s.split('_')[0] for s in df.columns])):
        vf=[col for col in df.columns if feature in col]
        if len(vf) > 1:
            vf_tilde=[]
            m1,m2=get_lowest(df,vf)
            vf_tilde.append(m1)
            vf_tilde.append(m2)
            vf.remove(m1)
            vf.remove(m2)
            while(len(vf)>0):
                # remove every m with argmax sim > p
                m_remove = []
                for m_test in vf:
                    max_sim = -1
                    for m in vf_tilde:
                        cos = cos_sim(df[m_test].values,df[m].values)
                        if cos > max_sim:
                            max_sim = cos
                    if max_sim > p:
                        m_remove.append(m_test)
                for m in m_remove:
                    vf.remove(m)
                # move m_test to vf_tilde with m_test = argmax_(m_i,m_j) cosine_sim(m_i,m_j)
                m_argmax = None
                max_sim = -1
                for m_test in vf:
                    for m in vf_tilde:
                        cos = cos_sim(df[m_test].values,df[m].values)
                        if cos > max_sim:
                            m_argmax = m_test
                            max_sim = cos
                if m_argmax:
                    vf_tilde.append(m_argmax)
                    vf.remove(m_argmax)
            features[feature]=vf_tilde
        else:
            if vf:
                features[feature].append(vf[0])
            else:
                raise NotImplementedError
    return features
# generate pool of similarity schemas
features=generate_pool(pairs,0.99)
features=dict(features)
a=list(features.values())
sim_schemas=list(list(schema) for schema in list(itertools.product(*a)))

### Jurek_Step 3 Seed selction using the generated features with field weighting
'''
inputs :
    df : the Dateframe containing the features
    features : the selected features
    Xm : list of matching seed
    Xu : list of unmatching seed
    w : features weights
outputs :
    w : new calculated weights
'''
def calculate_weights(df,measures,Xm,Xu,w):
    ls=[]
    djs=[]
    for index,m in enumerate(measures):
        match=abs(df.loc[Xm][m]-1).sum(axis=0)
        notmatch=abs(df.loc[Xu][m]-0).sum(axis=0)
        dj=match+notmatch
        djs.append(dj)
        if dj==0:
            ls.append(index)
    for index in range(w.shape[0]):
        if djs[index] == 0:
            wj = 1/len(ls)
        elif len(set(ls)-set([index])) >0:
            wj = 0
        else:
            wj=round(1/sum(djs[index]/d for d in djs),5)
        w[index]=wj
    return w
'''
inputs :
    df : the Dateframe containing the features
    Mm : maximum number of pairs to be selected as matching pairs
    Mu : maximum number of pairs to be selected as unmatching pairs
    e : weights difference threshold
    w : intial weights
outputs :
    Xm : selected matching seed as a set
    Xu : selected unmatching seed as a set
'''
def automatic_seed_selection(df,Mm,Mu,e,w):
    Xm=set()
    Xu=set()
    tm,tu=0,0
    while(len(Xm)<Mm and tm<1.1):
        tm+=0.05
        t=np.dot(abs(df[~df.index.isin(Xm)].values-1),w)
        Xm.update(df[~df.index.isin(Xm)][t<=tm].head(Mm-len(Xm)).index) ### fill the seed until we reach Mm without repeating
    while(len(Xu)<Mu and tu<1.1):
        tu+=0.05
        t=np.dot(abs(df[~df.index.isin(Xu.union(Xm))].values-0),w)
        Xu.update(df[(~df.index.isin(Xu.union(Xm)))][t<=tu].head(Mu-len(Xu)).index) ### make sure that no matching point is selected for not matching point
    wnew=calculate_weights(df,df.columns,Xm,Xu,w)
    while( np.array(abs(wnew-w)>e).any()):
        Xm=set()
        Xu=set()
        tm,tu=0,0
        w=wnew
        while(len(Xm)<Mm):
            tm+=0.05
            t=np.dot(abs(df[~df.index.isin(Xm)].values-1),w)
            Xm.update(df[~df.index.isin(Xm)][t<=tm].head(Mm-len(Xm)).index)
        while(len(Xu)<Mu):
            tu+=0.05
            t=np.dot(abs(df[~df.index.isin(Xu)].values-0),w)
            Xu.update(set(df[(~df.index.isin(Xu.union(Xm)))][t<=tu].head(Mu-len(Xu)).index))
        wnew=calculate_weights(df,df.columns,Xm,Xu,w)
    return Xm,Xu

Xms=[]
Xus=[]
for schema in sim_schemas:
    z=len(schema)
    w=np.full((z,1),1/z)
    Xm,Xu=automatic_seed_selection(pairs[list(schema)],num_duplicates_init_self_learning,num_non_duplicates_init_self_learning,0.5,w)
    Xms.append(Xm)
    Xus.append(Xu)


#### Jurek_step 4 select the most high diverse features based on their Q statistics value
'''
inputs : 
    set0 : selected intial seed 
    set1 : another selected intial seed 
outputs :
    Q : Q value for the two seeds
'''

def calculate_Q_matrix(S_tilde):
    num_sets = len(S_tilde)
    S = list(set.union(*S_tilde))
    Q_s = np.zeros((num_sets,num_sets))
    for i in range(num_sets):
        for j in range(i+1,num_sets):
            s00 = len([x for x in S if x not in S_tilde[i] and x not in S_tilde[j]])
            s10 = len([x for x in S if x in S_tilde[i] and x not in S_tilde[j]])
            s01 = len([x for x in S if x not in S_tilde[i] and x in S_tilde[j]])
            s11 = len([x for x in S if x in S_tilde[i] and x in S_tilde[j]])
            if (s00*s11+s01*s10) >0:
                Q_s[i][j] = (s00*s11-s01*s10)/(s00*s11+s01*s10)
    return Q_s

def calculate_average_Q_statistic(Q_matrix):
    sum_Q = 0
    L = Q_matrix.shape[0]
    for i in range(L-1):
        for j in range(i+1,L):
            sum_Q += Q_matrix[i][j]
    return (2*sum_Q)/(L*(L-1))

def select_sim_schemas_with_high_Q_statistic(sim_schemas,Xms,Xus,number_of_sets):
    S_tilde = list(Xms[index].union(Xus[index]) for index in range(len(Xms)))
    Q_s = calculate_Q_matrix(S_tilde)
    S_with_lowest_Q_statistic = S_tilde.copy()
    sim_schemas_lowest = sim_schemas.copy()
    Xms_lowest = Xms.copy()
    Xus_lowest = Xus.copy()
    while  len(S_with_lowest_Q_statistic) > number_of_sets:
        min_val = 1
        for index,seed_set in enumerate(S_with_lowest_Q_statistic):
            Q_s_without_index=np.delete(np.delete(Q_s,index,0),index,1)
            Q_avg = calculate_average_Q_statistic(Q_s_without_index)
            if min_val > Q_avg:
                min_val = Q_avg
                Q_s_new = Q_s_without_index.copy()
                S_lowest_new = [S_with_lowest_Q_statistic[ind] for ind \
                                in range(len(S_with_lowest_Q_statistic)) if ind != index]
                sim_schemas_lowest_new = [sim_schemas_lowest[ind] for ind \
                                in range(len(sim_schemas_lowest)) if ind != index]
                Xms_lowest_new = [Xms_lowest[ind] for ind \
                                in range(len(Xms_lowest)) if ind != index]
                Xus_lowest_new = [Xus_lowest[ind] for ind \
                                in range(len(Xus_lowest)) if ind != index]
        Q_s = Q_s_new.copy()
        S_with_lowest_Q_statistic = S_lowest_new.copy()
        sim_schemas_lowest = sim_schemas_lowest_new.copy()
        Xms_lowest = Xms_lowest_new.copy()
        Xus_lowest = Xus_lowest_new.copy()
    return sim_schemas_lowest,Xms_lowest,Xus_lowest


sim_schemas,Xms,Xus = select_sim_schemas_with_high_Q_statistic(sim_schemas,Xms,Xus,10)


#### Jurek_step 5 the self learning training process
models=[]
labels_per_schema=[]
probs_per_schema=[]
for i,schema in enumerate(sim_schemas):
    model=SelfLearningModel(LogisticRegression(tol=1e-3,solver='liblinear'))
    models.append(model)
    x = pairs[schema].values
    y_df = pd.DataFrame(list(-1 for i in range(x.shape[0])),columns=['y'],index=pairs.index)
    y_df.loc[Xms[i]]=1
    y_df.loc[Xus[i]]=0
    y = y_df['y'].values
    model.fit(x,y)
    labels = model.predict(x)
    labels_per_schema.append(labels)
    probs_per_schema.append(model.predict_proba(x)[:,1])

labels_ensemble = np.array([round(sum(list(labels_per_schema[i][j] \
                    for i in range(len(labels_per_schema)))) / len(labels_per_schema)) \
                   for j in range(len(labels_per_schema[0]))],int)

### Jurek_step 6 remove the classfiers that make a different predictions with other classfiers baes on a threshold
def calculate_CRs(labels_per_schema,labels_ensemble):
    CRs=[]
    for labels in labels_per_schema:
        CRs.append(np.average([labels==labels_ensemble]))
    return CRs
CRs=calculate_CRs(labels_per_schema,labels_ensemble)
CR_mean = np.average(CRs)

labels_per_schema_final = [labels_per_schema[i] for i in range(len(labels_per_schema)) \
                           if CRs[i]>=CR_mean]
probs_per_schema_final = [probs_per_schema[i] for i in range(len(labels_per_schema)) \
                           if CRs[i]>=CR_mean]

labels_ensemble_final = np.array([round(sum(list(labels_per_schema_final[i][j] \
                    for i in range(len(labels_per_schema_final)))) / len(labels_per_schema_final)) \
                   for j in range(len(labels_per_schema_final[0]))],int)

probs_ensemble_final = np.array([sum(list(probs_per_schema_final[i][j] \
                    for i in range(len(probs_per_schema_final)))) / len(probs_per_schema_final) \
                   for j in range(len(probs_per_schema_final[0]))],float)

# save restults of duplicate detection
pairs['label_jurek'] = labels_ensemble_final
pairs['probs_jurek'] = probs_ensemble_final
pairs.to_pickle(pairs_file_name.replace('.pkl','_with_labels.pkl'))
