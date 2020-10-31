#%%
import pandas as pd
import seaborn as sns
cleaned_data = pd.read_csv(r".\train_u6lujuX_CVtuZ9i.csv")
# made by sawan


# cleaned_data['ApplicantIncome'].plot.hist()
# plt.show()
sns.displot(cleaned_data['ApplicantIncome'])
sns.displot(cleaned_data['CoapplicantIncome'])
sns.displot(cleaned_data['LoanAmount'])
# %%
