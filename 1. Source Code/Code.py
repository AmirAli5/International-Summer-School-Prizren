"""
University of Prizren, Kosovo
                            International Summer Scool
                                                    Data Mining for Business Intelligence
                                                                                    Predicting Purchase Behavior in Social Network Ads
                                                                                                                                    Amir Ali
"""


######################################### Import Required Libraries ################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
plt.style.use("fivethirtyeight")

# Load the Dataset
df = pd.read_csv("Social_Network_Ads.csv")


######################################### Exploratory Data Analysis ################################################

# Gender vs Salary
plt.figure(figsize = (16, 9))
sns.scatterplot(data=df, x="Age", y="EstimatedSalary", hue = "Gender")

# gender distribution 1
dataset['Gender'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(6,6))
plt.title('Gender percentages', fontsize = 20)
plt.tight_layout()
plt.show()

# gender distribution 2
sns.countplot(dataset['Gender'], palette = 'Set2')
plt.title ('Gender vs. quantity', fontsize = 20)
plt.show()

# Age Distribution
sns.distplot(dataset['Age'], bins = 5, color = 'orange', label = 'KDE')
plt.legend()
plt.gcf().set_size_inches(12, 5)

# Age vs Gender
plt.figure(figsize = (22,10))
sns.countplot(x = 'Age',data = dataset , hue='Gender', palette = 'Set2')
plt.legend(loc='upper center')
plt.show()

# Purchased based on Age
sns.catplot(x="Age", col = 'Purchased', data=dataset, kind = 'count', palette='pastel')
plt.gcf().set_size_inches(20, 10)
#plt.gcf().autofmt_xdate()
plt.show()

# Purchased based on Gender
sns.catplot(x="Gender", col = 'Purchased', data=dataset, kind = 'count', palette='pastel')
plt.show()


######################################### Data Preprocessing ################################################

# Is there any missing value
df.isnull().sum()

# Handle Missing Value
age_imputer = SimpleImputer(strategy='mean')
df['Age'] = age_imputer.fit_transform(df[['Age']])

gender_mode = df['Gender'].mode()[0]
df['Gender'].fillna(gender_mode, inplace=True)

# Label Encoder
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Standard Scaler
from sklearn.preprocessing import StandardScaler
numerical_features = df[['Age', 'EstimatedSalary']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features)
df1 = pd.DataFrame(scaled_features, columns=['Age', 'Salary'])
gender_column = df['Gender']
purchased_columns = df['Purchased']
df1['Gender'] = gender_column
df1['Purchased'] = purchased_columns

# Define Label and Target Label
X = df.iloc[:, 1:-1]    
y = df.iloc[:, -1]

# Split the Data into Test and Train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Print the Shape of Training and Testing Data
print("Shape of Training Data", X_train.shape, y_train.shape)
print("Shape of Training Data", X_test.shape, y_test.shape)


######################################### Model Building ################################################
"""# 
1. Naive Bayes
2. Decision Tree
3. K-Nearest Neighbor
4. Support Vector Machine
5. Multilayer Perceptrons """

# Initialze the Model
DT =  DecisionTreeClassifier()
kNN =  KNeighborsClassifier()
MLPs = MLPClassifier()
SVM = SVC()
NB = GaussianNB()

# Fit the Training Data To Model
DT.fit(X_train, y_train)
kNN.fit(X_train, y_train)
MLPs.fit(X_train, y_train)
SVM.fit(X_train, y_train)
NB.fit(X_train, y_train)

# Make a Prediction
nb_pred = NB.predict(X_test)
svm_pred = SVM.predict(X_test)
knn_pred = kNN.predict(X_test)
dt_pred = DT.predict(X_test)
mlp_pred = MLPs.predict(X_test)

######################################### Result Evaluation ################################################

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, nb_pred)
print(cm)

# Accuracy Score
from sklearn.metrics import accuracy_score
acc_knn = accuracy_score(y_test, knn_pred)
acc_nb = accuracy_score(y_test, nb_pred)
acc_svm = accuracy_score(y_test, svm_pred)
acc_dt = accuracy_score(y_test, dt_pred)
acc_mlp = accuracy_score(y_test, mlp_pred)
print(acc_nb, acc_dt, acc_knn, acc_svm, acc_mlp)

# Precision
from sklearn.metrics import accuracy_score, precision_score
pr_knn = precision_score(y_test, knn_pred)
pr_nb = precision_score(y_test, nb_pred)
pr_svm = precision_score(y_test, svm_pred)
pr_dt = precision_score(y_test, dt_pred)
pr_mlp = precision_score(y_test, mlp_pred)

# F1 Score
f1_knn = f1_score(y_test, knn_pred)
f1_nb = f1_score(y_test, nb_pred)
f1_svm = f1_score(y_test, svm_pred)
f1_dt = f1_score(y_test, dt_pred)
f1_mlp = f1_score(y_test, mlp_pred)

# Recall
r_knn = recall_score(y_test, knn_pred)
r_nb = recall_score(y_test, nb_pred)
r_svm = recall_score(y_test, svm_pred)
r_dt = recall_score(y_test, dt_pred)
r_mlp = recall_score(y_test, mlp_pred)

######################################### END ################################################








