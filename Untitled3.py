#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('D:/Third Year/Internship/mushrooms.csv')
data


# In[4]:


pd.set_option('display.max_columns',None)


# In[5]:


data.head()


# In[6]:


data.tail()


# In[8]:


data.shape


# In[9]:


print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])


# In[11]:


data.info()


# In[15]:


data.isnull().sum()


# In[18]:


data.describe()
#statics of dataset


# In[19]:


#data Manipulation
data = data.astype('category')


# In[20]:


data.dtypes


# In[23]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in data.columns:
    data[column]=le.fit_transform(data[column])


# In[24]:


data.head()


# In[25]:


#feature matrix
X = data.drop('class',axis=1)         #independant variable
Y = data['class']                     #dependant variable


# In[26]:


X


# In[27]:


Y


# In[29]:


#applying principle component analysis
#for reducing dimensionality
from sklearn.decomposition import PCA
pca1 = PCA(n_components=7)
pca_fit = pca1.fit_transform(X)


# In[30]:


pca1.explained_variance_ratio_


# In[31]:


sum(pca1.explained_variance_ratio_)


# In[34]:


#splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(pca_fit,Y,test_size=0.30,random_state=42)


# In[35]:


X_train


# In[36]:


Y_train


# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#importing all important models


# In[40]:


#model training
lr = LogisticRegression()
lr.fit(X_train,Y_train)

knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)

svc = SVC()
svc.fit(X_train,Y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train,Y_train)


rm = RandomForestClassifier()
rm.fit(X_train,Y_train)

gb = GradientBoostingClassifier()
gb.fit(X_train,Y_train)


# In[41]:


#prediction on test data

Y_pred1 = lr.predict(X_test)
Y_pred2 = knn.predict(X_test)
Y_pred3 = svc.predict(X_test)
Y_pred4 = dt.predict(X_test)
Y_pred5 = rm.predict(X_test)
Y_pred6 = gb.predict(X_test)


# In[43]:


#evaluating algoritm

from sklearn.metrics import accuracy_score


# In[45]:


print("ACC LR",accuracy_score(Y_test,Y_pred1))
print("ACC knn",accuracy_score(Y_test,Y_pred2))
print("ACC svc",accuracy_score(Y_test,Y_pred3))
print("ACC dt",accuracy_score(Y_test,Y_pred4))
print("ACC rm",accuracy_score(Y_test,Y_pred5))
print("ACC gb",accuracy_score(Y_test,Y_pred6))


# In[47]:


final_data = pd.DataFrame({'Models':['LR','KNN','SVC','DT','RM','GB'],'ACC': [accuracy_score(Y_test,Y_pred1)*100,
                                                                accuracy_score(Y_test,Y_pred2)*100,
                                                                accuracy_score(Y_test,Y_pred3)*100,
                                                                accuracy_score(Y_test,Y_pred4)*100,
                                                                accuracy_score(Y_test,Y_pred5)*100,
                                                                accuracy_score(Y_test,Y_pred6)*100]})


# In[48]:


final_data


# In[49]:


import seaborn as sns


# In[51]:


sns.barplot(final_data['Models'],final_data['ACC'])

#RandomForest model is more accurate


# In[53]:


#save the model
rf_model = RandomForestClassifier()
rf_model.fit(pca_fit,Y)


# In[54]:


import joblib


# In[55]:


joblib.dump(rf_model,"Mushroom_prediction")


# In[56]:


model = joblib.load('Mushroom_prediction')


# In[60]:


p = model.predict(pca1.transform([[5,2,4,1,6,1,0,1,4,0,3,2,2,7,7,0,2,0,4,2,3,5]]))


# In[61]:


if p[0]==1:
     print('Poissonous')
else:
    print('Edible')


# In[62]:


from tkinter import *
import joblib


# In[71]:


def show_entry_fields():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=int(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=int(e7.get())
    p8=int(e8.get())
    p9=int(e9.get())
    p10=int(e10.get())
    p11=int(e11.get())
    
    p12=int(e12.get())
    p13=int(e13.get())
    p14=int(e14.get())
    p15=int(e15.get())
    p16=int(e16.get())
    p17=int(e17.get())
    p18=int(e18.get())
    p19=int(e19.get())
    p20=int(e20.get())
    p21=int(e21.get())
    p22=int(e22.get())
    
    model = joblib.load('Mushroom_prediction')
    result=model.predict(pca1.transform([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22]]))
    
    if result[0] == 0:
        Label(master,text="Edible").grid(row=31)
    else:
        Label(master,text = "Poissonous").grid(row=31)
        
master = Tk()
master.title("Mushroom Classification using Machine Learning")

label = Label(master, text = "Mushroom Classification using Machine Learning",bg = "white",fg = "black").    grid(row = 0,columnspan=2)

Label(master,text="cap-shape :(bell=0,conical=1,convex=5,flat=2,knobbed=3,sunken=4)").grid(row=1)
Label(master,text="cap-surface:(fibrous=0,grooves=1,scaly=3,smooth=2)").grid(row=2)
Label(master,text="cap-color:(brown=4,buff=0,cinnamon=1,green=9,pink=5,purple=6,red=2,white=7,yellow=8)").grid(row=3)
Label(master,text="bruises:(bruises=1,no=0)").grid(row=4)
Label(master,text="odor:(almond=0,anise=3,creosote=1,fishy=8,foul-2,musty=4,none=5,pungent=6,spicy=7)").grid(row=5)
Label(master,text="gill-attachment:(attached=0,descending=1,free=2,notched=3)").grid(row=6)
Label(master,text="gill-spacing:(close=0,crowded=2, distant=1)").grid(row=7)
Label(master,text="gill-size:(road=0,narrow=1)").grid(row=8)
Label(master,text="gill-color:(black-4,brown=5,buff=0, chocolate=3,gray=2,green=8,orange=6,pink=7,purple=9,red=1,white-10,yellow=11)").grid(row=9)
Label(master,text="stalk-shape: (enlarging=0,tapering=1)").grid(row=10)
Label(master,text="stalk-root:(bulbous-0,club=1,cup=5,equal=2, rhizomorphs=4, rooted=3,missing=6)").grid(row=11)
Label(master,text="stalk-surface-above-ring:(fibrous-0,scaly=3,silky=1,smooth=2)").grid(row=12)
Label(master,text="stalk-surface-below-ring:(fibrous=0, scaly=3,silky=1,smooth=2)").grid(row=13)
Label(master,text="stalk-color-above-ring:(brown=4,buff=0,cinnamon=1,gray=3, orange=5,pink=6,red=2,white=7,yellow=8)").grid(row=14)
Label(master,text="stalk-color-below-ring:(brown=4,buff=0,cinnamon=1,gray=3, orange=5,pink=6,red=2,white=7,yellow=8)").grid(row=15)
Label(master,text="veil-type:(partial=0,universal=1)").grid(row=16)
Label(master,text="veil-color:(brown=0,orange=1,white=2,yellow=3)").grid(row=17)
Label(master,text="ring-number:(none=0,one=1,two=2)").grid(row=18)
Label(master,text="ring-type:(cobwebby=0,evanescent-1,flaring-2,large-3,none-4,pendant=5,sheathing=6,zone=7)").grid(row=19)
Label(master,text="spore-print-color:(black=2,brown=3,buff=0,chocolate=1, green=5, orange=4,purple=6,white=7,yellow=8)").grid(row=20)
Label(master,text="population:(abundant=0,clustered=1,numerous=2,scattered=3, # several=4,solitary=5)").grid(row=21)
Label(master,text="habitat:(grasses=1,leaves=2,meadows=3,paths=4,urban=5,#waste=6,woods=0)").grid(row=22)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)
e13 = Entry(master)
e14 = Entry(master)
e15 = Entry(master)
e16 = Entry(master)
e17 = Entry(master)
e18 = Entry(master)
e19 = Entry(master)
e20 = Entry(master)
e21 = Entry(master)
e22 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)
e13.grid(row=13, column=1)
e14.grid(row=14, column=1)
e15.grid(row=15, column=1)
e16.grid(row=16, column=1)
e17.grid(row=17, column=1)
e18.grid(row=18, column=1)
e19.grid(row=19, column=1)
e20.grid(row=20, column=1)
e21.grid(row=21, column=1)
e22.grid(row=22, column=1)

Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()




# In[ ]:




