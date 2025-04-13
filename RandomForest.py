import os
import pandas as pd
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

os.getcwd()
os.chdir('C:\\Users\\bhargavjagan\\Downloads')
data = pd.read_csv('test.csv')
x = []
y = []
#dataset
for i in range(100):
    temp = data.loc[i][4]
    x.append(temp)
    pol = TextBlob(temp).sentiment.polarity
    if pol>0:
        pol = 1
    else:
        pol = 0
    y.append(pol)
    #print(i+1,'\n',temp,'\n',pol)
    #input()
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(x)

clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(clf,X,y,cv=5)
