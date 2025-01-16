#!/usr/bin/env python
# coding: utf-8

# In[117]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


# In[118]:


df=pd.read_csv("mail_data.csv")


# In[119]:


df.head()


# In[120]:


df.isnull().sum()


# In[121]:


df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})


# In[122]:


x=df["Message"]
y=df["Category"]


# In[123]:


x


# In[124]:


y


# In[125]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# In[126]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)


# In[127]:


len(x_train),len(x_test)


# In[128]:


len(y_train),len(y_test)


# In[129]:


feature_extract = TfidfVectorizer(stop_words='english', lowercase='True')
x_train = feature_extract.fit_transform(x_train)
x_test = feature_extract.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')


# In[131]:


model = MultinomialNB()

model.fit(x_train, y_train)

# Make predictions on the test se

accuracy1= accuracy_score(y_test, y_pred)*100
print("Accuracy of NaiveBayes:",accuracy1)


# In[ ]:





# In[102]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# In[132]:


model = LogisticRegression()

# Train the mode
model.fit(x_train, y_train)

# Make predictions on the test set


# Evaluate the model
accuracy2 = model.score(x_test,y_test)*100
print("Accuracy of LogisticRegression is:",accuracy2)


# In[133]:


svm = SVC(random_state=3)
svm.fit(x_train, y_train)
sc=svm.score(x_test,y_test)*100
print("Accuracy Of SVM is :",sc)


# In[134]:


RFC=RandomForestClassifier()
RFC.fit(x_train,y_train)
Score=RFC.score(x_test,y_test)*100
print("Accuracy of RnadomForestClassifier :",Score)


# In[153]:


import matplotlib.pyplot as plt

models = ['Naive Bayes','Logistic Regression','SVM', 'Random Forest']
accuracies = [accuracy1,accuracy2, sc, Score]

plt.figure(figsize=(10,9))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Models',fontsize=12)
plt.ylabel('Accuracy (%)',fontsize=12)
plt.title('Accuracy Comparison of Different Models',fontsize=14)
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()

for i, accuracy in enumerate(accuracies):
    plt.text(i, accuracy + 1, f'{accuracy:.2f}%', ha='center')

# Display the plot
plt.show()


# In[112]:


new_message = ["I've been searching for "]
new_message= feature_extract.transform(new_message)
prediction = model.predict(new_message)

if prediction == 1:
    print("Spam Mail")
else:
    print("Ham Mail")


# In[ ]:





# In[ ]:




