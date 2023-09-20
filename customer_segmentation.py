#%%
# Importing libraries
import os
import pickle
import datetime
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential, Input
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split 
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler 
from sklearn.metrics import confusion_matrix, classification_report 
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from module import missing_values, drop_columns, fill_missing_values, remove_rows

#%%
# path
CSV_PATH = os.path.join(os.getcwd(), 'data', 'Train.csv')
MODEL_PATH = os.path.join(os.getcwd(),'model.pkl')
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))

# %%
# 1.0 Data loading
df = pd.read_csv(CSV_PATH)

# %%
# 2.0 Data Inspecting
# Displaying 5 row of datasets
df.head()

# %%
# Checking info for the dataset
df.info()

#%%
# Check for shape of the dataset
shape_df = df.shape
print(f"The shape of the dataset: {shape_df}")

# %%
#Check duplicates in the dataset
duplicate = df[df.duplicated()]
print(f"Duplicate Rows : {duplicate.shape}")

#%%
# 3.0 Data Cleaning
# Handling Missing Value
missing_df = missing_values(df)

# %%
# Drop Columns 
# i) Dropping "days_since_prev_campaign_contact" columns from the dataset due to high percentage of missing value.
# ii) Dropping "id", "day_of_month" and "month" columns from the dataset since it is not applicable for this analysis
# Specify the columns to be dropped
columns_to_drop = ['days_since_prev_campaign_contact', 'id', 'day_of_month', 'month']

# Call the function to drop the specified columns
df = drop_columns(df, columns_to_drop)

# %%
# Substitute the missing values to relevant data
# For "marital" and "personal_loan" columns, the missing values will be replaced with "unknown"

# Specify fill values for specific columns in a dictionary
fill_values = {'marital': 'Unknown', 'personal_loan': 'Unknown'}

# Call the function to fill missing values in specified columns
df = fill_missing_values(df, fill_values)

# %%
# The other Missing values will be removed
# Specify the columns to consider for removing rows with missing values
columns_to_consider = ['customer_age', 'balance', 'last_contact_duration', 'num_contacts_in_campaign']

# Call the function to remove rows with missing values for the specified columns
df = remove_rows(df, columns_to_consider)

# %%
# Check if got any missing values anymore
missing_df = missing_values(df)

# %%
# 4.0 Data Visualization
# Separating between categorical columns and continuous columns
df_cont = df[['customer_age','balance', 'last_contact_duration', 'num_contacts_in_campaign']]
df_cat = df.drop(df_cont.columns, axis=1)

#%%
# Setting the color palette
sns.palplot(sns.color_palette("Accent"))
sns.set_palette("Accent")
sns.set_style('whitegrid')

#%%
# 4.1 Histogram for all continuous Variables
df_cont.hist(bins=50, layout=(3,3))
plt.tight_layout()

# %%
# 4.2 Countplot Vs 'term_deposit_subscribed' for all Categorical Variables

for column in df_cat.columns:
    plt.figure(figsize=(15,10))
    sns.countplot(x=column, data=df_cat, hue='term_deposit_subscribed')

    plt.title(f'{column} vs term_deposit_subscribed')

# %%
# 4.3 countplot for all cat cols
# Calculate the number of subplots needed based on the number of categorical columns
num_columns = len(df_cat.columns)
num_rows = (num_columns - 1) // 3 + 1  # Adjust the number of rows as needed
fig, axs = plt.subplots(num_rows, 3, figsize=(12, 12))
axs = axs.ravel()

# Loop through the categorical columns and create count plots
for i, column in enumerate(df_cat.columns):
    sns.countplot(x=column, data=df_cat, ax=axs[i])
    axs[i].set_title(f'{column}')
    
    # Rotate x-axis labels vertically
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)

# Remove any empty subplots if the number of columns is not a multiple of 3
for j in range(num_columns, num_rows * 3):
    fig.delaxes(axs[j])

# Adjust the layout
plt.tight_layout()
plt.show()
    
# %%
# 4.4 Correlation Matrix
corr = df_cont.corr() #calculate the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')

# %%
# 5.0 Feature Selection
# df['job_type']= pd.to_numeric(df['job_type'], errors = 'coerce')
# df['prev_campaign_outcome']= pd.to_numeric(df['prev_campaign_outcome'], errors = 'coerce')

X = df.drop(labels=['term_deposit_subscribed',
                    'marital','education','default','housing_loan',
                    'personal_loan','communication_type',
                    'num_contacts_in_campaign','job_type', 'prev_campaign_outcome'], axis =1)
# dropping term_deposit_subscribed because that supoose to be our target. 
# dropping 'marital','num_contacts_in_campaign','default','education' as they have very low correlational ratio

y = df['term_deposit_subscribed']

# %%
# 6.0 Data Preprocessing

def min_max_scaling(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_scaled = (X - X_min) / (X_max - X_min)
    return X_scaled

X = min_max_scaling(X)

#%%
ohe = OneHotEncoder(sparse = False)
y = ohe.fit_transform(np.array(y).reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                   random_state= 64)

#%%
# 7.0 Model Development
def build_simple_dl_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(num_classes, activation='softmax')
    ])
    
    model.summary()
    
    return model

#%% 
num_classes = len(np.unique(y_train, axis=0))
input_shape = X_train.shape[1:]

model = build_simple_dl_model(input_shape=input_shape, num_classes=num_classes)

#%%
# 8.0 Model Compilation 

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics = ['acc'])

tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)

early_callback = EarlyStopping(monitor = 'val_loss', patience=5) 

plot_model(model,show_shapes=(True))

#%%
# 9.0 Model Training
# List of callbacks
callbacks_list = [tensorboard_callback, early_callback]

# Training parameters
epoch_count = 50
validation_data_tuple = (X_test, y_test)

# Fit the model
history = model.fit(
    x=X_train, 
    y=y_train, 
    epochs=epoch_count,
    validation_data=validation_data_tuple,
    callbacks=callbacks_list
)

#%%
# 10.0 Model Evaluation
# Display the keys in the history object
print(f"History keys: {list(history.history.keys())}")

# Function to plot losses
def plot_losses(history_data):
    plt.figure()
    plt.plot(history_data.history['loss'], label='Train Loss')
    plt.plot(history_data.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

# Function to plot accuracies
def plot_accuracies(history_data):
    plt.figure()
    plt.plot(history_data.history['acc'], label='Train Accuracy')
    plt.plot(history_data.history['val_acc'], label='Validation Accuracy')
    plt.legend()
    plt.show()

# Plot losses and accuracies
plot_losses(history)
plot_accuracies(history)

# Evaluate the model on test data and display the score
test_score = model.evaluate(X_test, y_test)
print(f"Test score: {test_score}")
print("-----------------------------------------------------------------")
print(f"Best Accuracy score: {test_score[1]}")
print(f"Best Loss score: {test_score[0]}")

#%%
# 11.0 Model Analysis 
# Make predictions
predicted_probs = model.predict(X_test)

# Convert predicted probabilities to class labels
predicted_labels = np.argmax(predicted_probs, axis=1)

# Extract the true labels
actual_labels = np.argmax(y_test, axis=1)

# Generate confusion matrix and classification report
conf_matrix = confusion_matrix(actual_labels, predicted_labels)
class_report = classification_report(actual_labels, predicted_labels)

# Display the confusion matrix
label_names = ['Unsubscribe', 'Subscribe']
conf_matrix_display = ConfusionMatrixDisplay(conf_matrix, display_labels=label_names)
conf_matrix_display.plot(cmap='Blues')
plt.show()

# Print the classification report
print(f"Classification Report:\n{class_report}")


# %%
# 12. Saving and loading the model
# Save the model
model.save('customer_segmentation.h5')

# %%
