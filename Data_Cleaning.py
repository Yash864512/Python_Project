import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset ----------------------
file_path = "Air_Quality.csv"
df = pd.read_csv(file_path)

# Data Cleaning ----------------------
print("Missing values before handling:\n", df.isnull().sum(), "\n")

# Convert 'Data Value' to numeric (if not already)
df['Data Value'] = pd.to_numeric(df['Data Value'], errors='coerce')

# Fill missing values safely
df['Data Value'] = df['Data Value'].fillna(df['Data Value'].mean())
if 'Geo Place Name' in df.columns:
    df['Geo Place Name'] = df['Geo Place Name'].fillna(df['Geo Place Name'].mode()[0])
if 'Measure Info' in df.columns:
    df['Measure Info'] = df['Measure Info'].fillna("Unknown")

# Save cleaned file
df.to_csv("Air_Quality_Cleaned.csv", index=False)
print("Cleaned data saved to 'Air_Quality_Cleaned.csv'\n")

# Data Analysis ----------------------

# Dataset structure
print("Dataset Info:\n")
print(df.info())

# Summary statistics
print("\nSummary Statistics:\n", df.describe())

# Unique values in categorical columns
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    print(f"\nTop unique values in '{col}':")
    print(df[col].value_counts().head(10))

# Average Data Value by Measure
avg_by_measure = df.groupby('Measure')['Data Value'].mean().sort_values(ascending=False)
print("\nAverage Data Value by Measure:\n", avg_by_measure)

# Max Data Value by Geo Place Name
max_by_place = df.groupby('Geo Place Name')['Data Value'].max().sort_values(ascending=False).head(10)
print("\nTop 10 Geo Places by Max Data Value:\n", max_by_place)

# Convert to datetime if applicable
if 'Start_Date' in df.columns:
    df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')
    if pd.api.types.is_datetime64_any_dtype(df['Start_Date']):
        df['Year'] = df['Start_Date'].dt.year
        avg_by_year = df.groupby('Year')['Data Value'].mean()
        print("\nAverage Data Value by Year:\n", avg_by_year)
    else:
        print("'Start_Date' not in datetime format.")
else:
    print("'Start_Date' column not found.")

# Correlation Matrix
print("\nCorrelation Matrix:\n", df.select_dtypes(include=[np.number]).corr())

# Common Measure & Geo Type combinations
if 'Geo Type Name' in df.columns:
    top_combos = df.groupby(['Measure', 'Geo Type Name']).size().reset_index(name='Count')
    print("\nTop Measure & Geo Type Combinations:\n", top_combos.sort_values(by='Count', ascending=False).head(10))

# Visualizations ----------------------
sns.set(style="whitegrid")

# 1. Histogram with KDE
plt.figure(figsize=(10, 6))
sns.histplot(df['Data Value'], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Data Value")
plt.xlabel("Data Value")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2. Boxplot by Measure
plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x='Measure', y='Data Value', hue='Measure', legend=False, palette='Set3')
plt.title("Boxplot of Data Value by Measure")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Violin Plot
plt.figure(figsize=(14, 6))
sns.violinplot(data=df, x='Measure', y='Data Value', hue='Measure', legend=False, palette='Pastel1')
plt.title("Violin Plot of Data Value by Measure")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Top 10 Geo Places
top_geo = df['Geo Place Name'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(y=top_geo.index, x=top_geo.values, hue=top_geo.index, dodge=False, legend=False, palette='Blues')
plt.title("Top 10 Geo Place Names by Frequency")
plt.xlabel("Count")
plt.ylabel("Geo Place Name")
plt.tight_layout()
plt.show()

# 5. Countplot of Geo Type Name
if 'Geo Type Name' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, y='Geo Type Name', hue='Geo Type Name', dodge=False, legend=False,
                  order=df['Geo Type Name'].value_counts().index, palette='coolwarm')
    plt.title("Count of Geo Type Name")
    plt.tight_layout()
    plt.show()

# 6. Time Series Plot
if 'Start_Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Start_Date']):
    time_data = df.groupby('Start_Date')['Data Value'].mean().reset_index()
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=time_data, x='Start_Date', y='Data Value', color='green')
    plt.title("Average Data Value Over Time")
    plt.xlabel("Start Date")
    plt.ylabel("Average Data Value")
    plt.tight_layout()
    plt.show()

# 7. Pie Chart
if 'Geo Type Name' in df.columns:
    geo_counts = df['Geo Type Name'].value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(geo_counts, labels=geo_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title("Geo Type Distribution")
    plt.tight_layout()
    plt.show()

# 8. Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()

# 9. Barplot of Average by Measure
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_by_measure.values, y=avg_by_measure.index, hue=avg_by_measure.index, dodge=False, legend=False, palette='Set2')
plt.title("Average Data Value by Measure")
plt.xlabel("Average Data Value")
plt.ylabel("Measure")
plt.tight_layout()
plt.show()

# Save Final Transformed File ----------------------
df.to_csv("Air_Quality_Transformed.csv", index=False)
print("\nTransformed data saved to 'Air_Quality_Transformed.csv'")
