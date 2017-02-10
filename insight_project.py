import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,roc_curve,auc
from scipy import spatial
from sklearn.preprocessing import StandardScaler

def find_d(g):
  thres = 0.8
  g = g.sort_values(by='distance2', ascending=True)
  g['cumsum_trans'] = g['Transaction'].cumsum()
  g['cumsum_trans'] = g['cumsum_trans']/(g['cumsum_trans'].iloc[-1])
  idx = [x[0] for x in enumerate(g['cumsum_trans']) if x[1]>=thres][0]
  return g.iloc[idx][['distance2','Latitude_s','Longitude_s']]

def find_d_list(df):
  newdf = df.groupby('StoreID').apply(find_d)
  return newdf
  
def find_da_list_within_d(newdf):
  res = []
  for _, row in newdf.iterrows():
	  r = tree.query_ball_point((row['Latitude_s'], row['Longitude_s']), row['distance2'])
	  res.append(r)
  return res

def array_to_index(res):
  dfs = []
  for i, l in enumerate(res):
	df = pd.DataFrame({'daid':l}, index=[i]*len(l))
	dfs.append(df)
  dfs = pd.concat(dfs)

  storeid = map(lambda x: storeid_list[x], dfs.index)
  dauid   = DA_cord.iloc[dfs.daid]['DAUID']
  return pd.DataFrame({'StoreID': storeid, 'DAUID': dauid.values})
 
def store_features_df(res,store,DA,sumlist,avglist):
  # dauid to store map
  storeid_dauid_map = array_to_index(res)

# merge store, sales, pop
  merge = pd.merge(pd.merge(storeid_dauid_map,store[['Sales','StoreID']],on='StoreID'),DA[['P_TOTPOPSUM','DAUID']],on='DAUID')

# groupby store, sum pop, mean sales
  tmp = merge.groupby('StoreID')[['P_TOTPOPSUM','Sales']].agg({'P_TOTPOPSUM':np.sum, 'Sales':np.mean})
  tmp['score'] = tmp.Sales/tmp.P_TOTPOPSUM
  tmp = tmp.sort_values(by='score',ascending=False)

# storeid merged with da
  storeid_da = storeid_dauid_map.merge(DA,on='DAUID') 
  ops={key:np.sum for key in sumlist}
  ops.update({key: np.mean for key in avglist})
  store_da_features = storeid_da.groupby('StoreID')[sumlist+avglist].agg(ops)
  store_da_score = store_da_features.join(tmp[['P_TOTPOPSUM','score']])
  store_da_score=store_da_score.sort_values(by='score',ascending=False)

# calcualte % of pop
  for i in sumlist:
	store_da_score[i] = store_da_score[i]/store_da_score['P_TOTPOPSUM'] 
  return store_da_score

def select_top_poor(store_da_score,num):
  nums = int(340*num)
  top_poor=pd.concat([store_da_score[:nums],store_da_score[-nums:]],axis=0)
  label = [1,0]*nums
  label.sort(reverse=True)
  top_poor['label'] = label
  return top_poor

def prepare_train(store_da_score, percentage):
  num = int(296 * percentage)
  poor_shuffle = store_da_score.iloc[44:,:].sample(frac=1,random_state=42)
  poor = poor_shuffle[:num]
  top = store_da_score.iloc[:44,:]
  train = pd.concat([top,poor],axis=0)
  label = [1]*44 + [0]*num
  train['label']=label
  train = train.dropna()
  X_train = train.iloc[:,:3]
  y_train = train.label
  return X_train, y_train

def logit_model(X_train, y_train):

  # normalize
  scaler = StandardScaler().fit(X_train)
  X_train_scale=scaler.transform(X_train)
  logit = sm.Logit(y_train, X_train_scale)
  sol = logit.fit()
  return sol,X_train_scale

def newDA(DA2):
  DA = DA2.copy()
  #age groups  
  DA['P_PT09'] = DA['P_PT04'] + DA['P_PT59']
  DA['P_PT1024'] = DA['P_PT1014'] + DA['P_PT1519'] + DA['P_PT2024']
  DA['P_PT2549'] = DA['P_PT2529'] + DA['P_PT3034'] +  DA['P_PT3539'] + DA['P_PT4044'] + DA['P_PT4549']
  DA['P_PT5064'] = DA['P_PT5054'] + DA['P_PT5559'] + DA['P_PT6064']
  DA['P_PT65P'] = DA['P_PT6569'] + DA['P_PT7074'] + DA['P_PT7579'] + DA['P_PT8084'] + DA['P_PT85P']

  # income groups
  DA['P_INC0_40P'] = DA['P_INC10P'] + DA['P_INC10_20'] + DA['P_INC20_30'] + DA['P_INC30_40']
  DA['P_INC40_90'] = DA['P_INC40_50'] + DA['P_INC50_60'] + DA['P_INC60_70'] + DA['P_INC70_80'] + DA['P_INC80_90'] 
  DA['P_INC90P'] = DA['P_INC90_100'] + DA['P_INC100_125'] + DA['P_INC125_150'] + DA['P_INC150P']
  
  # housing year groups
  DA['P_DBB6111'] = DA['P_DBB6180'] + DA['P_DBB8190'] +  DA['P_DBB9100'] + DA['P_DBB0105'] + DA['P_DBB0611']

# drop orignal age and income lists
  Drop_Income = ['P_INC10P','P_INC10_20','P_INC20_30','P_INC30_40','P_INC40_50','P_INC50_60','P_INC60_70','P_INC70_80','P_INC80_90','P_INC90_100','P_INC100_125','P_INC125_150','P_INC150P']
  Drop_Age = ['P_PT04','P_PT59','P_PT1014','P_PT1519','P_PT2024','P_PT2529','P_PT3034','P_PT3539','P_PT4044','P_PT4549','P_PT5054','P_PT5559','P_PT6064','P_PT6569','P_PT7074','P_PT7579','P_PT8084','P_PT85P']
  Drop_DBB = ['P_DBB6180', 'P_DBB8190', 'P_DBB9100', 'P_DBB0105', 'P_DBB0611']
  
  DA.drop(Drop_Income, axis=1,inplace=True)
  DA.drop(Drop_Age,axis=1,inplace=True)
  DA.drop(Drop_DBB, axis=1, inplace=True)

  return DA


def select_features(sumlist,avglist,percentage):
  DA2 = newDA(DA)

  #df with store and da 
  store_da_score = store_features_df(res,store,DA2,sumlist,avglist)
  
  # choose % for training
  top_poor = select_top_poor(store_da_score,percentage)
  top_poor = top_poor.dropna()
  
  X_train = top_poor.iloc[:,:-3]
  y_train = top_poor.label
  
  # logistic regression
  sol,X_train_scale = logit_model(X_train,y_train)
  tvalues = pd.DataFrame({'features': X_train.columns,'tvalues': sol.tvalues})
  print tvalues 
  return store_da_score

if __name__ == '__main__':
  
  result = pd.read_csv('v1_result.csv')
  DA = pd.read_csv('Canada_DA_Demographics_new.csv')
  store=pd.read_csv('StoreLocations_Canada.csv')
  DA_cord = pd.read_csv('DA_Centroids_withCoords.csv')
  #storeid_dauid_map=pd.read_csv('storeid_dauid_map.csv')
  
  # calculate cord distance
  result['distance2'] = np.sqrt((result['Latitude_c']-result['Latitude_s'])**2+(result['Longitude_c']-result['Longitude_s'])**2)

  # find dauids that contributes to 80% sales
  tree = spatial.KDTree(zip(DA_cord.Latitude.ravel(),DA_cord.Longitude.ravel()))
  newdf = find_d_list(result)
  res=find_da_list_within_d(newdf)
  storeid_list = sorted(result.StoreID.unique())
  
  #storeid_dauid_map = array_to_index(res)

  # features selection using logistic regression
  sumlist = ['P_EDNONE', 'P_PT65P']
  avglist = ['P_EVEGET']
  store_da_score = select_features(sumlist,avglist,0.13)
  
  # cluster stores using score
  # reason to use 13% data to select feature
  km = KMeans(n_clusters=2)
  km.fit(store_da_score.score.values.reshape(-1,1))
  #len(km.labels_[km.labels_==1])

  # use top 13% and all rest data to train 
  X_train, y_train = prepare_train(store_da_score, 1.0)

  # model
  clf = RandomForestClassifier(n_estimators=60,random_state=42)
  solution_rf = clf.fit(X_train,y_train)
  score=cross_val_score(clf,X_train,y_train,cv=3).mean()
  print "The cross validation score is %f" % (score)
