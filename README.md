# SpamDetection
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
data=pd.read_csv("spam.csv")
print(data.head())
print(data.shape)
data.drop_duplicates(inplace=True)
print(data.shape)
print(data.isnull().sum())
data['Category']=data['Category'].replace(['ham','spam'],['Not Spam','Spam'])
print(data.head())
mess=data['Message']
cat=data['Category']
(mess_train,mess_test,cat_train,cat_test)=train_test_split(mess,cat,test_size=0.2)
cv=CountVectorizer(stop_words='english')
features=cv.fit_transform(mess_train)

#Creating Model

model=MultinomialNB()
model.fit(features,cat_train)

#Test our Model
features_test=cv.transform(mess_test)
print(model.score(features_test,cat_test))

#predict Data
message=cv.transform(['congratulations,you won a lottery']).toarray()
result=model.predict(message)
print(result)

message1=cv.transform(['hey we have meeting tomorrow']).toarray()
result=model.predict(message1)
print(result)
