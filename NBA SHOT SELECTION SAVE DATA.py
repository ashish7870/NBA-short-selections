#!/usr/bin/env python
# coding: utf-8

# # This dataset appears to be a log of basketball shots taken during a game, 
# capturing various details about each shot. Here's a breakdown of the key columns and their descriptions:
# 
# 1.action_type:
#     
#     Describes the type of shot or play action (e.g., Jump Shot, Driving Dunk Shot).
# 
# 2.combined_shot_type: 
# 
#     A more generalized shot type (e.g., Jump Shot, Dunk).
# 
# 3.game_event_id:
#     
#     A unique identifier for the event within a specific game.
# 
# 4.game_id:
#         
#         A unique identifier for the game.
# 5.lat: 
#     Latitude coordinate of the shot on the court.
# 6.loc_x:
#     
#     X-coordinate on the court.
# 
# 7.loc_y:
#         Y-coordinate on the court.
# 8.lon:
#     
#     Longitude coordinate of the shot on the court.
# 
# 9.minutes_remaining: 
#     Minutes remaining in the current period.
# 
# 10.period: 
#         The period of the game (e.g., 1 for first quarter).
# 
# 11.shot_type: 
#     
#     Specifies if the shot is a 2-point or 3-point field goal.
# 
# 12.shot_zone_area: 
#     
#     Describes the area of the court from which the shot was taken (e.g., Right Side, Left Side Center).
# 
# 13.shot_zone_basic: 
#     
#     A basic classification of the shot's location (e.g., Mid-Range, Restricted Area).
# 
# 14.shot_zone_range: 
#     
#     Describes the range of the shot (e.g., 16-24 ft., Less Than 8 ft.).
# 
# 15.team_id: 
#    
#    A unique identifier for the team.
# 
# 16.team_name:
#     
#     The name of the team.
# 
# 17.game_date: 
#         
#         The date of the game.
# 
#         
# 18.matchup: 
#     
#     The matchup of the game (e.g., LAL @ POR).
# 
# 19.opponent: 
#     
#     The opposing team.
# 
# 20.shot_id:
#    
#    A unique identifier for the shot.
# 
# 
# This dataset can be used to analyze shot patterns, player performance, and game strategies by looking at the spatial and temporal distribution of shots taken during a basketball game.
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np 


# # Basic check

# In[80]:


df = pd.read_csv("NBA.csv")


# In[81]:


df.head()


# In[82]:


df['shot_made_flag'].value_counts()


# In[83]:


df


# # datapreprocessing

# In[84]:


df = df.drop(labels = ["matchup", "game_date", "team_name", "team_id", "seconds_remaining", "game_event_id", "game_id"],axis = 1)


# In[85]:


df.info()


# In[86]:


df['season'].value_counts()


# In[87]:


df.head()


# # EDa

# In[88]:


# Filter shots data
jump_shots = df[df['combined_shot_type'] == "Jump Shot"]
other_shots = df[df['combined_shot_type'] != "Jump Shot"]

# Create the plot
plt.figure(figsize=(8, 6))

# Plot jump shots with grey color and alpha 0.3
plt.scatter(jump_shots['lon'], jump_shots['lat'], color='grey', alpha=0.2, label='Jump Shot')

# Plot other shots with different colors and alpha 0.8
sns.scatterplot(data=other_shots, x='lon', y='lat', hue='combined_shot_type', alpha=0.8, palette='deep')

# Set plot title and labels
plt.title('Shot type', fontsize=15)
plt.ylim(33.7, 34.0883)
plt.xlabel('')
plt.ylabel('')
plt.legend(title='')
plt.axis('off')
plt.show()


# In[89]:


df.info()


# In[90]:


from matplotlib.gridspec import GridSpec

# Plot 1: Shot zone range scatter plot
plt.figure(figsize=(8, 8))
gs = GridSpec(2, 1, height_ratios=[2, 1])

# Subplot 1
ax1 = plt.subplot(gs[0])
sns.scatterplot(data=df, x='lon', y='lat', hue='shot_zone_range', palette='deep', ax=ax1)
ax1.set_title('Shot zone range', fontsize=15)
ax1.set_ylim(33.7, 34.0883)
ax1.set_xlabel('')
ax1.set_ylabel('')
if ax1.legend_ is not None:
    ax1.legend_.remove()
ax1.axis('off')

# Plot 2: Frequency for each shot zone range
ax2 = plt.subplot(gs[1])
sns.countplot(data=df, x='shot_zone_range', order=df['shot_zone_range'].value_counts().index, palette='deep', ax=ax2)
ax2.set_ylabel('Number of Shots')
ax2.set_xlabel('')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.grid(True, axis='y', linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()


# In[91]:


# Plot 1: Shot zone area scatter plot
plt.figure(figsize=(8, 8))
gs = GridSpec(2, 1, height_ratios=[2, 1])

# Subplot 1
ax1 = plt.subplot(gs[0])
sns.scatterplot(data=df, x='lon', y='lat', hue='shot_zone_area', palette='deep', ax=ax1)
ax1.set_title('Shot zone area', fontsize=15)
ax1.set_ylim(33.7, 34.0883)
ax1.set_xlabel('')
ax1.set_ylabel('')
if ax1.legend_ is not None:
    ax1.legend_.remove()
ax1.axis('off')

# Plot 2: Frequency for each shot zone area
ax2 = plt.subplot(gs[1])
sns.countplot(data=df, x='shot_zone_area', order=df['shot_zone_area'].value_counts().index, palette='deep', ax=ax2)
ax2.set_ylabel('Number of Shots')
ax2.set_xlabel('')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, fontsize=7)
ax2.grid(True, axis='y', linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()


# In[92]:


# Plot 1: Shot zone basic scatter plot
plt.figure(figsize=(8, 8))
gs = GridSpec(2, 1, height_ratios=[2, 1])

# Subplot 1
ax1 = plt.subplot(gs[0])
sns.scatterplot(data=df, x='lon', y='lat', hue='shot_zone_basic', palette='deep', ax=ax1)
ax1.set_title('Shot zone basic', fontsize=15)
ax1.set_ylim(33.7, 34.0883)
ax1.set_xlabel('')
ax1.set_ylabel('')
if ax1.legend_ is not None:
    ax1.legend_.remove()
ax1.axis('off')

# Plot 2: Frequency for each shot zone basic
ax2 = plt.subplot(gs[1])
sns.countplot(data=df, x='shot_zone_basic', order=df['shot_zone_basic'].value_counts().index, palette='deep', ax=ax2)
ax2.set_ylabel('Number of Shots')
ax2.set_xlabel('')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, fontsize=6.3)
ax2.grid(True, axis='y', linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()



# In[93]:


# Group by action_type and calculate accuracy and counts
grouped_df = df.groupby('action_type').agg(
    Accuracy=('shot_made_flag', 'mean'),
    counts=('shot_made_flag', 'size')
).reset_index()

# Filter out action types with counts <= 20
filtered_df = grouped_df[grouped_df['counts'] > 20]

# Sort by ascending accuracy
filtered_df = filtered_df.sort_values(by='Accuracy')

# Create the plot
plt.figure(figsize=(8, 6))
ax = sns.scatterplot(
    data=filtered_df,
    x='Accuracy',
    y='action_type',
    hue='Accuracy',
    size='Accuracy',
    sizes=(100, 300),  # Increase the size of the dots
    palette=sns.color_palette("RdYlGn", as_cmap=True),
    legend=False
)
# Customize the plot
ax.set_title("Accuracy by shot type", fontsize=15)
ax.set_ylabel('')
ax.set_xlabel('Shot Accuracy')
plt.gca().invert_yaxis()
plt.grid(True, axis='x', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()


# In[94]:


# Group by season and calculate mean accuracy
accuracy_by_season = df.groupby('season')['shot_made_flag'].mean().reset_index()

# Plotting
plt.figure(figsize=(8, 6))
ax = sns.lineplot(
    data=accuracy_by_season,
    x='season',
    y='shot_made_flag',
    marker='.',
    color='grey',
    legend=False,
    zorder=1
)
ax = sns.scatterplot(
    data=accuracy_by_season,
    x='season',
    y='shot_made_flag',
    s=150,
    color='purple',
    legend=False,
    zorder=2
)

plt.title('Accuracy by Season', fontsize=15)
plt.xlabel('Season')
plt.ylabel('Shot Accuracy')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()

    


# In[95]:


df.info()


# In[96]:


# Calculate mean accuracy by opponent
accuracy_by_opponent = df.groupby('opponent')['shot_made_flag'].mean().reset_index()

# Define conference for each opponent
conference_mapping = {
    "Eastern": ["ATL", "BOS", "BRK", "CHI", "CHO", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"],
    "Western": ["DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA"]
}
accuracy_by_opponent['Conference'] = accuracy_by_opponent['opponent'].apply(lambda x: 'Eastern' if x in conference_mapping['Eastern'] else 'Western')
# Reorder opponents by accuracy
accuracy_by_opponent = accuracy_by_opponent.sort_values(by='shot_made_flag', ascending=False)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
width = 0.7
x = np.arange(len(accuracy_by_opponent))

eastern_mask = accuracy_by_opponent['Conference'] == 'Eastern'
western_mask = accuracy_by_opponent['Conference'] == 'Western'

ax.bar(x[eastern_mask], accuracy_by_opponent[eastern_mask]['shot_made_flag'], width, color='lightblue', linewidth=0, label='Eastern')
ax.bar(x[western_mask], accuracy_by_opponent[western_mask]['shot_made_flag'], width, color='lightpink', linewidth=0, label='Western')

ax.set_title('Accuracy by Opponent')
ax.set_xlabel('Opponent')
ax.set_ylabel('Shot Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(accuracy_by_opponent['opponent'], rotation=45, ha='right')
ax.legend(title='Conference')

plt.tight_layout()
plt.show()


# In[97]:


sns.countplot(x="shot_made_flag",data = df)


# In[98]:


df.info()


# In[99]:


# Calculate accuracy by opponent for 2PT Field Goal and 3PT Field Goal shots
accuracy_by_opponent_shot_type = df.groupby(['opponent', 'shot_type'])['shot_made_flag'].mean().reset_index()
accuracy_by_opponent_shot_type = accuracy_by_opponent_shot_type[accuracy_by_opponent_shot_type['shot_type'].isin(['2PT Field Goal', '3PT Field Goal'])]
accuracy_by_opponent_shot_type = accuracy_by_opponent_shot_type.pivot(index='opponent', columns='shot_type', values='shot_made_flag')
accuracy_by_opponent_shot_type.columns = ['TwoPoint', 'ThreePoint']

# Reorder opponents based on the order in the previous graph
opponent_order = accuracy_by_opponent.index
accuracy_by_opponent_shot_type = accuracy_by_opponent_shot_type.reindex(opponent_order)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
x = range(len(accuracy_by_opponent_shot_type))

ax.plot(x, accuracy_by_opponent_shot_type['TwoPoint'], marker='o', markersize=8, color='lightblue', label='2PT Field Goal')
ax.plot(x, accuracy_by_opponent_shot_type['ThreePoint'], marker='o', markersize=8, color='lightpink', label='3PT Field Goal')

ax.set_title('Accuracy by Opponent', fontsize=16, ha='center')
ax.set_xlabel('Opponent', fontsize=12)
ax.set_ylabel('Shot Accuracy', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(accuracy_by_opponent_shot_type.index, rotation=45, ha='right', fontsize=10)
ax.legend(loc='lower center', ncol=2, fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


# In[100]:


# Filter shots for 2PT and 3PT Field Goals
shots_2pt = df[df['shot_type'] == "2PT Field Goal"]
shots_3pt = df[df['shot_type'] == "3PT Field Goal"]

# Calculate mean accuracy by season for 2PT and 3PT Field Goals separately
accuracy_2pt = shots_2pt.groupby('season')['shot_made_flag'].mean().reset_index()
accuracy_3pt = shots_3pt.groupby('season')['shot_made_flag'].mean().reset_index()

# Create a categorical variable for seasons
season_order = sorted(df['season'].unique())
accuracy_2pt['season'] = pd.Categorical(accuracy_2pt['season'], categories=season_order, ordered=True)
accuracy_3pt['season'] = pd.Categorical(accuracy_3pt['season'], categories=season_order, ordered=True)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Plot 2PT Field Goal accuracy
ax.plot(accuracy_2pt['season'], accuracy_2pt['shot_made_flag'], marker='o', markersize=8, color='lightblue', label='2PT Field Goal')

# Plot 3PT Field Goal accuracy
ax.plot(accuracy_3pt['season'], accuracy_3pt['shot_made_flag'], marker='o', markersize=8, color='lightpink', label='3PT Field Goal')

ax.set_title('Accuracy by Season', fontsize=16, ha='center')
ax.set_xlabel('Season', fontsize=12)
ax.set_ylabel('Shot Accuracy', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax.legend(loc='lower center', ncol=2, fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


# In[101]:


df.head(10)


# In[102]:


# Group by shot_distance and calculate mean accuracy
accuracy_by_distance = df.groupby('shot_distance')['shot_made_flag'].mean().reset_index()

# Plotting
plt.figure(figsize=(8, 6))

# Scatter plot
sns.scatterplot(
    data=accuracy_by_distance,
    x='shot_distance',
    y='shot_made_flag',
    s=150,
    color='purple',
    legend=False,
    zorder=2  # Set the order of scatter plot above line plot
)

# Line plot
sns.lineplot(
    data=accuracy_by_distance,
    x='shot_distance',
    y='shot_made_flag',
    marker='.',
    color='black',
    legend=False,
    zorder=1  # Set the order of line plot below scatter plot
)

plt.title('Accuracy by Shot Distance', fontsize=15)
plt.xlabel('Shot Distance (ft.)')
plt.ylabel('Shot Accuracy')
plt.xlim(0, 45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[103]:


df.info()


# In[104]:


sns.countplot(x="shot_made_flag",data = df)


# In[105]:


df.shot_made_flag.value_counts()


# In[106]:


pip install imbalanced-learn


# In[107]:


df.isnull().sum()


# In[123]:


mode_value=df['shot_made_flag'].mode()[0]
df['shot_made_flag'].fillna(mode_value,inplace=True)


# In[124]:


df.isnull().sum()


# In[125]:


df.shot_made_flag.value_counts()


# In[126]:


df.info()


# # ENCODING

# In[127]:


# Encode categorical variables
label_encoders = {}
categorical_columns = ['action_type', 'combined_shot_type', 'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'opponent', 'season']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])


# In[147]:


df.head()


# ## spilittin the terain test data

# In[148]:


from sklearn.model_selection import train_test_split
df1 = df.drop(columns=['shot_made_flag'],axis=1)


# In[150]:


from sklearn.preprocessing import MinMaxScaler
import pandas as pd



# Create a MinMaxScaler instance
ms = MinMaxScaler()

# Fit the scaler to the data and transform it
X = ms.fit_transform(df1)

# If you want to convert it back to a DataFrame
X = pd.DataFrame(df1_scaled, columns=df1.columns)

print(df1_scaled)


# In[154]:


y = df["shot_made_flag"]


# In[151]:


X.head()


# # logistic Regression

# In[156]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)


# In[158]:


X_train.shape


# In[159]:


y_train.shape


# In[161]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,y_train)


# In[165]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
y_pred = lr.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))


# # Random Forest classifier

# In[167]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# Initialize Random Forest classifier
model = RandomForestClassifier(random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report for detailed performance metrics
print(classification_report(y_test, y_pred))

# Optionally, you can also inspect feature importances if using Random Forest
if isinstance(model, RandomForestClassifier):
    feature_importances = pd.DataFrame(model.feature_importances_,
                                       index = X.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    print("Feature Importances:\n", feature_importances)



# # XGB

# In[168]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Define parameter grid
param_grid = {
    'learning_rate': [0.1, 0.01],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize XGBClassifier
xgb_model = XGBClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model,
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# Predict on test data with best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))


# # hyperparameter Random Forest

# In[170]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Define parameter distributions for RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf_model,
                                   param_distributions=param_dist,
                                   n_iter=10,
                                   cv=5,
                                   scoring='accuracy',
                                   n_jobs=-1,
                                   random_state=42)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters:", random_search.best_params_)
print("Best CV Accuracy:", random_search.best_score_)

# Predict on test data with best model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))


# # mlp Ann

# In[172]:


import tensorflow as tf
from tensorflow import keras


# In[175]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt




# Initialize and train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(200,), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
y_prob = mlp.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
print(f"ROC AUC: {roc_auc}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# # Svm

# In[181]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt



# Initialize and train the SVM model
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
y_prob = svm.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
print(f"ROC AUC: {roc_auc}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# # Model Overview: Basketball Shot Analysis
# 1. Data Loading and Inspection:
# 
# Load the dataset containing information about basketball shots.
# Inspect the columns, check for missing values, and ensure data integrity.
# 
# 2. Data Cleaning and Preparation:
# 
# 3. Handle missing or Nan value.
# 
# Convert data types as necessary (e.g., timestamps, numerical values).
# Encode categorical variables if needed (e.g., shot types, shot zones).
# 
# 4. Exploratory Data Analysis (EDA):
# 
# Calculate basic statistics (e.g., shot counts, shooting percentages).
# Visualize shot distributions on the basketball court using heatmaps or scatter plots.
# Explore correlations between variables (e.g., shot success rates by shot type or location).
# Feature Engineering:
# 
# Create new features if beneficial (e.g., distance from the basket, angles of shots).
# Aggregate data at different levels (e.g., player-level statistics, team-level statistics).
# 
# 5. Model Building:
# 
# 6. Classification Task (Predicting Shot Outcome):
# 
# Train a classification model (e.g., Logistic Regression, Random Forest, Neural Network) to predict whether a shot is successful or not based on features such as shot location, type, game context (period, time remaining), etc.
# Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.
# 
# 7. Regression Task (Predicting Shot Efficiency):
# Predict shot efficiency metrics such as shooting percentage or effective field goal percentage.
# Use regression models (e.g., Linear Regression, Decision Trees) and evaluate using metrics like mean squared error or R-squared.
# Model Evaluation and Optimization:
# 
# 8. Perform cross-validation to assess model generalization.
# Tune hyperparameters to optimize model performance.
# Consider feature selection or dimensionality reduction techniques if necessary.
# Insights and Interpretation:
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# # I tried several times but the accuracy did not get better. Almost I tried all the good models, still did not get good accuracy
# 

# In[ ]:




