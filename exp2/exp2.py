import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  # For statistical analysis

# URLs for the datasets
train_url = "https://raw.githubusercontent.com/bodysoda2022/dataset-2024/refs/heads/main/2a.%20loan-train.csv"
test_url = "https://raw.githubusercontent.com/bodysoda2022/dataset-2024/refs/heads/main/2b.%20loan-test.csv"

# Directly load datasets from the URLs
df = pd.read_csv(train_url)

# Display the first few rows
print(df.head())

# Descriptive statistics
print(df.describe())

# Correlation heatmap
numerical_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(8, 6))
corr_matrix = numerical_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='inferno', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Gender vs Married
pd.crosstab(df.Gender, df.Married).plot(kind="bar", stacked=True, figsize=(5, 5), color=['#f64f59', '#12c2e9'])
plt.title('Gender vs Married')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()

# Self Employed vs Credit History
pd.crosstab(df.Self_Employed, df.Credit_History).plot(kind="bar", stacked=True, figsize=(5, 5),
                                                      color=['#544a7d', '#ffd452'])
plt.title('Self Employed vs Credit History')
plt.xlabel('Self Employed')
plt.ylabel('Frequency')
plt.legend(["Bad Credit", "Good Credit"])
plt.xticks(rotation=0)
plt.show()

# Property Area vs Loan Status
pd.crosstab(df.Property_Area, df.Loan_Status).plot(kind="bar", stacked=True, figsize=(5, 5),
                                                   color=['#333333', '#dd1818'])
plt.title('Property Area vs Loan Status')
plt.xlabel('Property Area')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()

# Applicant Income vs Co-applicant Income
df.plot(x='ApplicantIncome', y='CoapplicantIncome', style='o')
plt.title('Applicant Income - Co Applicant Income')
plt.xlabel('ApplicantIncome')
plt.ylabel('CoapplicantIncome')
plt.show()

# Pearson Correlation
print('Pearson correlation:', df['ApplicantIncome'].corr(df['CoapplicantIncome']))

# T-Test
print('T Test and P value: \n', stats.ttest_ind(df['ApplicantIncome'], df['CoapplicantIncome']))

# Check missing values
print(df.isnull().sum())

# Fill missing values
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df = df.drop(['Loan_ID'], axis=1)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)

# Verify missing values are handled
print(df.isnull().sum())

# Remove outliers using IQR
numerical_df = df.select_dtypes(include=['float64', 'int64'])
Q1 = numerical_df.quantile(0.25)
Q3 = numerical_df.quantile(0.75)
IQR = Q3 - Q1
df_cleaned = numerical_df[~((numerical_df < (Q1 - 1.5 * IQR)) | (numerical_df > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df_cleaned)

# Transform variables
df['ApplicantIncome'] = np.sqrt(df['ApplicantIncome'])
df['CoapplicantIncome'] = np.sqrt(df['CoapplicantIncome'])
df['LoanAmount'] = np.sqrt(df['LoanAmount'])

# Visualizations
sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
sns.histplot(data=df, x="ApplicantIncome", kde=True, ax=axs[0, 0], color='green')
sns.histplot(data=df, x="CoapplicantIncome", kde=True, ax=axs[0, 1], color='skyblue')
sns.histplot(data=df, x="LoanAmount", kde=True, ax=axs[1, 0], color='orange')
plt.tight_layout()
plt.show()
