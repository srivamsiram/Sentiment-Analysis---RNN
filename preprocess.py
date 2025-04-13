import pandas as pd
import os
import re
from string import punctuation
from textblob import TextBlob
import csv
os.chdir('F:\\Projects\\Main Project\\Code\\main')
data = pd.read_csv('mobile_dataset.csv')
text ="bhargav ihaodihi iugiug 8687 87698_)9*)&**(&56"
def preprocess(text):
    for c in text:
            t =ord(c)
            if not ((t < 123 and t > 96) or (t<91 and t>64) or t==32):
                text=text.replace(c,'')
    text = re.split(r'\W+',text)
    words = [word.lower() for word in text]
    text = " ".join(word for word in words)
    return text
reviews = data['Reviews']
count = 0
with open('result.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile,dialect="excel")
    n = int(input(len(reviews)))       
    for i in range(n):
        if pd.notnull(reviews[i]):            
            temp = preprocess(reviews[i])
            sent = TextBlob(temp)
            if sent.polarity >= 0:
                temp1 = 1            
            else:
                temp1 = 0        
            count = count +1
            writer.writerow([temp,temp1])
        else:
            print(i)
print(i,count)
