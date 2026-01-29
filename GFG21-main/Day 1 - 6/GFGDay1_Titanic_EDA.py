import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

# Set plot style for better aesthetics
sns.set(style='whitegrid')

# ---
# ### Step 1: Data Loading and Initial Inspection
# Load the dataset
titanic_df = pd.read_csv('https://raw.githubusercontent.com/GeeksforGeeksDS/21-Days-21-Projects-Dataset/master/Datasets/Titanic-Dataset.csv')

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(titanic_df.head())

# Get a concise summary of the dataframe
print("\nDataset Information:")
titanic_df.info()

# Get descriptive statistics for numerical columns
print("\nDescriptive Statistics:")
print(titanic_df.describe())

# ---
# ### Step 2: Data Cleaning
print("Missing values before cleaning:")
print(titanic_df.isna().sum())

# Handle missing 'Age' values with the median
median_age = titanic_df['Age'].median()
titanic_df['Age'].fillna(median_age, inplace=True)

# Handle missing 'Embarked' values with the mode
mode_embarked = titanic_df['Embarked'].mode()[0]
titanic_df['Embarked'].fillna(mode_embarked, inplace=True)

# Create a 'Has_Cabin' feature and drop the original 'Cabin' column
titanic_df['Has_Cabin'] = titanic_df['Cabin'].notna().astype(int)
titanic_df.drop('Cabin', axis=1, inplace=True)

# Verify that there are no more missing values
print("\nMissing values after cleaning:")
print(titanic_df.isna().sum())

# ---
# ### Step 3: Univariate Analysis
print("\nUnivariate Analysis of Categorical Features:")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Univariate Analysis of Categorical Features', fontsize=16)

sns.countplot(ax=axes[0, 0], x='Survived', data=titanic_df).set_title('Survival Distribution')
sns.countplot(ax=axes[0, 1], x='Pclass', data=titanic_df).set_title('Passenger Class Distribution')
sns.countplot(ax=axes[0, 2], x='Sex', data=titanic_df).set_title('Gender Distribution')
sns.countplot(ax=axes[1, 0], x='Embarked', data=titanic_df).set_title('Port of Embarkation')
sns.countplot(ax=axes[1, 1], x='SibSp', data=titanic_df).set_title('Siblings/Spouses Aboard')
sns.countplot(ax=axes[1, 2], x='Parch', data=titanic_df).set_title('Parents/Children Aboard')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\nUnivariate Analysis of Numerical Features:")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Univariate Analysis of Numerical Features', fontsize=16)

sns.histplot(ax=axes[0], data=titanic_df, x='Age', kde=True, bins=30).set_title('Age Distribution')
sns.histplot(ax=axes[1], data=titanic_df, x='Fare', kde=True, bins=40).set_title('Fare Distribution')

plt.show()

# ---
# ### Step 4: Bivariate Analysis
print("\nBivariate Analysis: Feature vs. Survival")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Bivariate Analysis with Survival', fontsize=16)

sns.barplot(ax=axes[0, 0], x='Pclass', y='Survived', data=titanic_df).set_title('Survival Rate by Pclass')
sns.barplot(ax=axes[0, 1], x='Sex', y='Survived', data=titanic_df).set_title('Survival Rate by Sex')
sns.barplot(ax=axes[1, 0], x='Embarked', y='Survived', data=titanic_df).set_title('Survival Rate by Port')
sns.barplot(ax=axes[1, 1], x='Has_Cabin', y='Survived', data=titanic_df).set_title('Survival Rate by Cabin Availability')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Age vs. Survival using a FacetGrid
g = sns.FacetGrid(titanic_df, col='Survived', height=6)
g.map(sns.histplot, 'Age', bins=25, kde=True)
plt.suptitle('Age Distribution by Survival Status', y=1.02)
plt.show()

# ---
# ### Step 5: Feature Engineering
titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1
titanic_df['IsAlone'] = (titanic_df['FamilySize'] == 1).astype(int)

print("\nCreated 'FamilySize' and 'IsAlone' features:")
print(titanic_df[['FamilySize', 'IsAlone']].head())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(ax=axes[0], x='FamilySize', y='Survived', data=titanic_df).set_title('Survival Rate by Family Size')
sns.barplot(ax=axes[1], x='IsAlone', y='Survived', data=titanic_df).set_title('Survival Rate for Those Traveling Alone')
plt.show()

# Extract 'Title' from 'Name'
titanic_df['Title'] = titanic_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
titanic_df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare', inplace=True)
titanic_df['Title'].replace(['Mlle', 'Ms', 'Mme'], 'Miss', inplace=True)

plt.figure(figsize=(12, 6))
sns.barplot(x='Title', y='Survived', data=titanic_df)
plt.title('Survival Rate by Title')
plt.show()

# ---
# ### Step 6: Multivariate Analysis
sns.catplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df, kind='bar', height=6, aspect=1.5)
plt.title('Survival Rate by Pclass and Sex')
plt.show()

# Violin plot of Age by Sex and Survival
plt.figure(figsize=(14, 8))
sns.violinplot(x='Sex', y='Age', hue='Survived', data=titanic_df, split=True, palette={0: 'blue', 1: 'orange'})
plt.title('Age Distribution by Sex and Survival')
plt.show()

# ---
# ### Step 7: Correlation Analysis
plt.figure(figsize=(14, 10))
numeric_cols = titanic_df.select_dtypes(include=np.number)
correlation_matrix = numeric_cols.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# ---
# ### Step 8: Generating ydata-profiling Report
# Generate the profiling report for the cleaned dataset
profile = ProfileReport(titanic_df, title="Titanic Dataset Profiling Report")

# Save the report to an HTML file
profile.to_file("titanic_eda_report.html")
print("\n'titanic_eda_report.html' has been saved.")