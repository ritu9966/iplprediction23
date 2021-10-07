import pandas as pd
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
deliveries=pd.read_csv('/home/stellamarsh/ipl_project/ipl/deliveries.csv',engine='python')
data=pd.read_csv("/home/stellamarsh/ipl_project/ipl/matches.csv",engine='python')


#print(data.head())
#print(data.shape)

#print(data.tail())
#print(data.describe())

#sb.countplot(data['winner'])

#plt.xticks(rotation=90)
#batting_first=data[data['win_by_runs']!=0]
#plt.figure(figsize=(7,7))
#plt.pie(list(batting_first['winner'].value_counts()),labels=list(batting_first['winner'].value_counts().keys()),autopct='%0.1f%%')

#plt.show()


#batting_second=data[data['win_by_wickets']!=0]

#plt.figure(figsize=(7,7))

#plt.pie(list(batting_second['winner'].value_counts()),labels=list(batting_second['winner'].value_counts().keys()),autopct='%0.1f%%')

#plt.show()


#most_runs=deliveries.groupby(['batsman','batting_team'])['batsman_runs'].sum().sort_values(ascending=False).reset_index().head(10)


#runs=sb.barplot(x="batsman",y="batsman_runs",data=most_runs,edgecolor=(0,0,0))
#runs.set_ylabel('Total Runs')
#runs.set_xlabel('Batsman')

#plt.xticks(rotation=90)
#plt.title("Total Runs per Batsman")
#plt.show()

new_data=data[['team1','team2','toss_decision','toss_winner','winner']]

new_data.dropna(inplace=True)

all_teams={}

ct=0


for i in range(len(data)):
    if data.loc[i]['team1'] not in all_teams:
        all_teams[data.loc[i]['team1']]=ct
        ct=ct+1

    if data.loc[i]['team2'] not in all_teams:
        all_teams[data.loc[i]['team2']]=ct
        ct=ct+1


x=new_data[['team1','team2','toss_decision','toss_winner']]
y=new_data[['winner']]

encoded_teams={w:k for k,w in all_teams.items()}


x=np.array(x)
y=np.array(y)

for i in range(len(x)):
    x[i][0]=all_teams[x[i][0]]
    x[i][1]=all_teams[x[i][1]]
    x[i][3]=all_teams[x[i][3]]

    y[i][0]=all_teams[y[i][0]]


fb={'field':0,'bat':1}

for i in range(len(x)):
    x[i][2]=fb[x[i][2]]


for i in range(len(x)):
    if x[i][3]==x[i][0]:
        x[i][3]=0
    else:
        x[i][3]=1



ones=0
for i in range(len(y)):
    if y[i][0]==x[i][1]:
        if ones<370:
            ones+=1
            y[i][0]=1
        else:
            t=x[i][1]
            x[i][0]=x[i][1]
            x[i][1]=t
            y[i][0]=0

    else:
        y[i][0]=0

x=np.array(x,dtype='int32')
y=np.array(y,dtype='int32')
y=y.ravel()

print(np.unique(y,return_counts=True))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


model1=SVC().fit(x_train,y_train)
model1.score(x_test,y_test)

model2=DecisionTreeClassifier().fit(x_train,y_train)
model2.score(x_test,y_test)

model3=RandomForestClassifier(n_estimators=200).fit(x_train,y_train)

model3.score(x_test,y_test)
test=np.array([1,2,0,0]).reshape(1,-1)
print(model1.predict(test))
print(model2.predict(test))
print(model3.predict(test))

with open('/home/stellamarsh/ipl_project/ipl/model1.pkl','wb') as f:
    pkl.dump(model3,f)

with open('/home/stellamarsh/ipl_project/ipl/vocab.pkl','wb') as f:
    pkl.dump(encoded_teams,f)
with open('/home/stellamarsh/ipl_project/ipl/inv_vocab.pkl','wb') as f:
    pkl.dump(all_teams,f)

