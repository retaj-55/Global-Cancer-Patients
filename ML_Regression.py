import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor


# Load the dataset
data = pd.read_csv('Original data/global_cancer_patients_2015_2024.csv')

## Data Exploration and Visualization

# Display basic info
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Visualize distribution of categorical features
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='Gender', data=data)
plt.title('Gender Distribution')

plt.subplot(2, 2, 2)
sns.countplot(x='Cancer_Type', data=data, order=data['Cancer_Type'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Cancer Type Distribution')

plt.subplot(2, 2, 3)
sns.countplot(x='Cancer_Stage', data=data, order=data['Cancer_Stage'].value_counts().index)
plt.title('Cancer Stage Distribution')

plt.subplot(2, 2, 4)
sns.countplot(x='Country_Region', data=data, order=data['Country_Region'].value_counts().iloc[:10].index)
plt.xticks(rotation=45)
plt.title('Top 10 Country Distribution')

plt.tight_layout()
plt.savefig('data_distributions.png')
plt.show()

# Visualize numerical features
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Age Distribution')

plt.subplot(2, 3, 2)
sns.histplot(data['Genetic_Risk'], bins=20, kde=True)
plt.title('Genetic Risk Distribution')

plt.subplot(2, 3, 3)
sns.histplot(data['Treatment_Cost_USD'], bins=20, kde=True)
plt.title('Treatment Cost Distribution')

plt.subplot(2, 3, 4)
sns.histplot(data['Survival_Years'], bins=20, kde=True)
plt.title('Survival Years Distribution')

plt.subplot(2, 3, 5)
sns.histplot(data['Target_Severity_Score'], bins=20, kde=True)
plt.title('Severity Score Distribution')

plt.tight_layout()
plt.savefig('numerical_distributions.png')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
# Update your correlation matrix code to:
corr = data.select_dtypes(include=['number']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()

# First ensure Target_Severity_Score is numeric
data['Target_Severity_Score'] = pd.to_numeric(data['Target_Severity_Score'], errors='coerce')

# Select numerical features
numerical_features = ['Age', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 
                     'Smoking', 'Obesity_Level', 'Treatment_Cost_USD', 'Survival_Years']

# Create pairplot with updated parameters
plt.figure(figsize=(20, 20))
pairplot = sns.pairplot(
    data=data[numerical_features + ['Target_Severity_Score']].dropna(),
    hue='Target_Severity_Score', 
    palette='viridis',
    plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'},
    diag_kind='hist',
    diag_kws={'alpha': 0.7, 'bins': 20}
)

# Adjust title and layout
pairplot.fig.suptitle('Feature Relationships with Severity Score', y=1.02)
plt.tight_layout()
plt.savefig('feature_pairplot.png', bbox_inches='tight')
plt.show()

## Data Preprocessing

# Convert categorical variables to numerical
label_encoders = {}
categorical_cols = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and target
X = data.drop(['Patient_ID', 'Target_Severity_Score'], axis=1)
y = data['Target_Severity_Score']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['Age', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 
                 'Smoking', 'Obesity_Level', 'Year', 'Treatment_Cost_USD', 'Survival_Years']

X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Save preprocessed data
X_train.to_csv('Preprocessed data/X.csv', index=False)
X_test.to_csv('Preprocessed data/X_test.csv', index=False)
y_train.to_csv('Preprocessed data/Y.csv', index=False)
y_test.to_csv('Preprocessed data/Y_test.csv', index=False)

## Model Training and Evaluation

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'ANN': MLPRegressor(max_iter=1000, random_state=42)
}

results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'model': model,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred
    }
    
    # Save predictions
    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    pred_df.to_csv(f'Results/prediction_{name.replace(" ", "")}_Model.csv', index=False)
    
    print(f"\n{name} Results:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

# Plot actual vs predicted for all models
plt.figure(figsize=(20, 15))
plt.suptitle('Actual vs Predicted Severity Scores', y=1.02, fontsize=16)

n_rows = 2
n_cols = 3

for i, (name, result) in enumerate(results.items()):
    plt.subplot(n_rows, n_cols, i+1)
    plt.scatter(y_test, result['predictions'], alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{name}\nR2: {result["r2"]:.2f}, RMSE: {result["rmse"]:.2f}')

# Remove empty subplots
for i in range(len(results), n_rows*n_cols):
    plt.subplot(n_rows, n_cols, i+1)
    plt.axis('off')

plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()

# Compare model performances
model_names = list(results.keys())
r2_scores = [results[name]['r2'] for name in model_names]
rmse_scores = [results[name]['rmse'] for name in model_names]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# R2 scores plot
sns.barplot(x=model_names, y=r2_scores, hue=model_names, 
            ax=ax1, palette='viridis', legend=False)
ax1.set_title('Model R2 Score Comparison')
ax1.set_ylabel('R2 Score')
ax1.set_ylim(0, 1)

# RMSE scores plot
sns.barplot(x=model_names, y=rmse_scores, hue=model_names,
            ax=ax2, palette='viridis', legend=False)
ax2.set_title('Model RMSE Comparison')
ax2.set_ylabel('RMSE')

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# Create a DataFrame of model metrics
metrics = []
for name, result in results.items():
    metrics.append({
        'Model': name,
        'MAE': result['mae'],
        'MSE': result['mse'],
        'RMSE': result['rmse'],
        'R2 Score': result['r2']
    })

metrics_df = pd.DataFrame(metrics).set_index('Model')

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(metrics_df.T, 
            annot=True, 
            fmt=".2f", 
            cmap="YlGnBu",
            linewidths=.5,
            cbar_kws={'label': 'Score'})

plt.title('Model Performance Comparison Heatmap', pad=20, fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('model_performance_heatmap.png', bbox_inches='tight', dpi=300)
plt.show()