#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
cleaned_data = pd.read_csv(r".\train_u6lujuX_CVtuZ9i.csv")

# made by sawan
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.subplot(231)
sns.countplot(x="Gender", hue='Loan_Status', data=cleaned_data)
plt.subplot(232)
sns.countplot(x="Married", hue='Loan_Status', data=cleaned_data)
plt.subplot(233)
sns.countplot(x="Education", hue='Loan_Status', data=cleaned_data)
plt.subplot(234)
sns.countplot(x="Dependents", hue='Loan_Status', data=cleaned_data)
plt.subplot(235)
sns.countplot(x="Property_Area", hue='Loan_Status', data=cleaned_data)

# made by sawan
# %%
