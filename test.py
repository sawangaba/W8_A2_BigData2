#%%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from dtreeplt import dtreeplt
from sklearn.metrics import accuracy_score


data1 = pd.read_csv(r".\train_u6lujuX_CVtuZ9i.csv")
data2= pd.read_csv(r".\test_Y3wMUE5_7gLdaTN.csv")


#data cleaning: replacing null values with most occuring and median values.
#Data 1 Cleaning
data1['Gender'].fillna(data1['Gender'].mode()[0], inplace=True)
data1['Married'].fillna(data1['Married'].mode()[0], inplace=True)
data1['Dependents'].fillna(data1['Dependents'].mode()[0], inplace=True)
data1['Self_Employed'].fillna(data1['Self_Employed'].mode()[0], inplace=True)
data1['Credit_History'].fillna(data1['Credit_History'].mode()[0], inplace=True)

data1['LoanAmount'].fillna(data1['LoanAmount'].median(), inplace=True)
data1['Loan_Amount_Term'].fillna(data1['Loan_Amount_Term'].mode()[0], inplace=True)


#Data 2 Cleaning
data2['Gender'].fillna(data2['Gender'].mode()[0], inplace=True)
data2['Married'].fillna(data2['Married'].mode()[0], inplace=True)
data2['Dependents'].fillna(data2['Dependents'].mode()[0], inplace=True)
data2['Self_Employed'].fillna(data2['Self_Employed'].mode()[0], inplace=True)
data2['Credit_History'].fillna(data2['Credit_History'].mode()[0], inplace=True)

data2['LoanAmount'].fillna(data2['LoanAmount'].median(), inplace=True)
data2['Loan_Amount_Term'].fillna(data2['Loan_Amount_Term'].mode()[0], inplace=True)


train= data1.drop('Loan_ID', axis=1)
test= data2.drop('Loan_ID', axis=1)
train['Dependents'].replace('3+',3,inplace=True)
test['Dependents'].replace('3+',3,inplace=True)
train['Loan_Status'].replace('N', 0,inplace=True)
train['Loan_Status'].replace('Y', 1,inplace=True)
test = test.drop(["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"],axis=1)
train = train.drop(["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"],axis=1)

ff=['Credit_History','Gender_Female','Gender_Male','Married_No','Married_Yes','Dependents_0','Dependents_1','Dependents_2','Dependents_3+','Education_Graduate','Education_Not Graduate','Self_Employed_No','Self_Employed_Yes','Property_Area_Rural','Property_Area_Semiurban','Property_Area_Urban']

X=train.drop("Loan_Status",1)
y=train[["Loan_Status"]]
X = pd.get_dummies(X)

train=pd.get_dummies(train)
test=pd.get_dummies(test)
x_train,x_cv,y_train,y_cv=train_test_split(X,y,test_size=0.3,random_state=1)
tree_model = DecisionTreeClassifier(random_state=1)
tree_model.fit(x_train,y_train)
pred_cv_tree=tree_model.predict(x_cv)

score_tree =accuracy_score(pred_cv_tree,y_cv)*100 
score_tree
pred_test_tree = tree_model.predict(test)

#plot the decision tree using dtree library

# dtree = dtreeplt(
#     model=pred_test_tree,
#     feature_names=ff,
#     target_names= ['0','1']
# )
# fig = dtree.view()
# fig.savefig('decision_tree.png')


# made by sawan
# %%
