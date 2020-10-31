#%%
import pandas as pd
import matplotlib.pyplot as plt
# made by sawan
plt.rcParams["figure.figsize"] = [10, 5]

cleaned_data = pd.read_csv(r".\train_u6lujuX_CVtuZ9i.csv")


print("Query 1. Find out the number of male and female in loan applicants data: \n")
print(cleaned_data['Gender'].value_counts())
print("\n")
cleaned_data['Gender'].value_counts(normalize=True).plot(kind='bar', color='blue')
plt.show()


print("Query 2. Find out the number of married and unmarried loan applicants: \n")
print(cleaned_data['Married'].value_counts())
print("\n")
cleaned_data['Married'].value_counts(normalize=True).plot(kind='bar', color='red')
plt.show()

print("Query 3. Find out the overall dependent status in the dataset: \n")
print(cleaned_data['Dependents'].value_counts())
print("\n")
cleaned_data['Dependents'].value_counts(normalize=True).plot(kind='bar', color='orange')
plt.show()


print("Query 4. Find the count how many loan applicants are graduate and non graduate: \n")
print(cleaned_data['Education'].value_counts())
print("\n")
cleaned_data['Education'].value_counts(normalize=True).plot(kind='bar', color='black')
plt.show()


print("Query 5. Find out the count how many loan applicants property lies in urban, rural and semi-urban areas: \n")
print(cleaned_data['Property_Area'].value_counts())
print("\n")
cleaned_data['Property_Area'].value_counts(normalize=True).plot(kind='bar', color='green')
plt.show()


# made by sawan

# %%
