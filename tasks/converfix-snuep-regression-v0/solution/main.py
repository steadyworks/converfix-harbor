import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os

df = pd.read_csv("/home/data/train.csv")
df_test = pd.read_csv("/home/data/test.csv")


df.head()


df.tail()


df.describe()


df.isnull().sum()


df.duplicated().sum()


df.shape


df.dtypes


df.nunique()


df.columns


# ## Data visualizations


df['last_login_date'] = pd.to_datetime(df['last_login_date'])


plt.figure(figsize=(8,5))
plt.hist(df['user_engagement_score'], bins=40, color='navy', alpha=0.75)

plt.title("Distribution of User Engagement Score")
plt.xlabel("Engagement Score")
plt.ylabel("User Count")
plt.show()



app_engagement = df.groupby('app_name')['user_engagement_score'].mean()

plt.figure(figsize=(8,5))
plt.bar(app_engagement.index, app_engagement.values,
        color=['#ff6f61', '#6b5b95', '#88b04b'])

plt.title("Average Engagement Score by App")
plt.xlabel("App Name")
plt.ylabel("Avg Engagement Score")
plt.show()



time_cols = [
    'time_on_feed_per_day',
    'time_on_explore_per_day',
    'time_on_messages_per_day',
    'time_on_reels_per_day'
]

time_mean = df[time_cols].mean()

plt.figure(figsize=(8,5))
plt.bar(time_mean.index, time_mean.values,
        color=['orange', 'gold', 'tomato', 'purple'])

plt.title("Average Daily Time Spent by Feature")
plt.ylabel("Minutes per Day")
plt.xticks(rotation=30)
plt.show()



plt.figure(figsize=(8,5))
plt.scatter(df['following_count'], df['followers_count'],
            color='green', alpha=0.5)

plt.title("Followers vs Following")
plt.xlabel("Following Count")
plt.ylabel("Followers Count")
plt.show()



plt.figure(figsize=(8,5))
plt.scatter(df['age'], df['user_engagement_score'],
            color='purple', alpha=0.5)

plt.title("Age vs User Engagement")
plt.xlabel("Age")
plt.ylabel("Engagement Score")
plt.show()



plt.figure(figsize=(8,5))
plt.boxplot(df['user_engagement_score'],
            patch_artist=True,
            boxprops=dict(facecolor='lightblue'))

plt.title("Engagement Score Distribution")
plt.ylabel("Engagement Score")
plt.show()



gender_engagement = df.groupby('gender')['user_engagement_score'].mean()

plt.figure(figsize=(6,4))
plt.bar(gender_engagement.index, gender_engagement.values,
        color=['pink', 'skyblue'])

plt.title("Average Engagement by Gender")
plt.xlabel("Gender")
plt.ylabel("Engagement Score")
plt.show()



content_eng = (
    df.groupby('content_type_preference')['user_engagement_score']
    .mean()
    .sort_values()
)

plt.figure(figsize=(8,5))
plt.barh(content_eng.index, content_eng.values, color='teal')

plt.title("Engagement by Content Type Preference")
plt.xlabel("Avg Engagement Score")
plt.show()



plt.figure(figsize=(8,5))
plt.plot(df['notification_response_rate'],
         df['user_engagement_score'],
         'o', color='darkred', alpha=0.4)

plt.title("Notification Response Rate vs Engagement")
plt.xlabel("Notification Response Rate")
plt.ylabel("Engagement Score")
plt.show()



# ## Feature engg


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


if 'last_login_date' in df.columns:
    df['last_login_date'] = pd.to_datetime(df['last_login_date'], errors='coerce')
    df['last_login_year'] = df['last_login_date'].dt.year
    df['last_login_month'] = df['last_login_date'].dt.month
    df.drop(columns=['last_login_date'], inplace=True)

if 'last_login_date' in df_test.columns:
    df_test['last_login_date'] = pd.to_datetime(df_test['last_login_date'], errors='coerce')
    df_test['last_login_year'] = df_test['last_login_date'].dt.year
    df_test['last_login_month'] = df_test['last_login_date'].dt.month
    df_test.drop(columns=['last_login_date'], inplace=True)



# Store test user_ids for submission
test_user_ids = df_test['user_id'].copy()


yes_no_cols = [
    'has_children',
    'uses_premium_features',
    'two_factor_auth_enabled',
    'biometric_login_used'
]

for col in yes_no_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    if col in df_test.columns:
        df_test[col] = df_test[col].map({'Yes': 1, 'No': 0})



cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    le = LabelEncoder()
    # Fit on combined train+test to handle all categories
    combined = pd.concat([df[col], df_test[col] if col in df_test.columns else pd.Series()], ignore_index=True)
    le.fit(combined.astype(str))
    df[col] = le.transform(df[col].astype(str))
    if col in df_test.columns:
        df_test[col] = le.transform(df_test[col].astype(str))



X = df.drop(columns=[
    'user_id',
    'user_engagement_score'
])

y = df['user_engagement_score']

# Prepare test features
X_test_final = df_test.drop(columns=['user_id'])



print("Non-numeric columns left:", X.select_dtypes(include='object').columns)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)



models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

mse_scores = {}
r2_scores = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)

    mse_scores[name] = mean_squared_error(y_val, y_pred)
    r2_scores[name] = r2_score(y_val, y_pred)

    print(f"{name} - MSE: {mse_scores[name]:.4f}, R2: {r2_scores[name]:.4f}")


# Select best model based on lowest MSE
best_model_name = min(mse_scores, key=mse_scores.get)
print(f"\nBest model: {best_model_name}")

# Retrain best model on full training data
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X)
X_test_scaled = scaler_full.transform(X_test_final)

best_model = models[best_model_name]
best_model.fit(X_full_scaled, y)

# Make predictions on test set
test_predictions = best_model.predict(X_test_scaled)


# Create submission file
submission = pd.DataFrame({
    'user_id': test_user_ids,
    'user_engagement_score': test_predictions
})
submission.to_csv('/home/submission/submission.csv', index=False)
print("Submission saved to /home/submission/submission.csv")
print(submission.head())


plt.figure(figsize=(12,6))
plt.bar(mse_scores.keys(), mse_scores.values())
plt.title("ML Model MSE Comparison")
plt.ylabel("Mean Squared Error")
plt.xticks(rotation=30)
plt.show()


