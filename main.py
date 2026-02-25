import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (limited rows for speed)
df = pd.read_csv(r'C:\Users\ASUS\Downloads\Fraud.csv', nrows=200000)

print("Shape:", df.shape)
print(df.head())

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Check outliers in amount
sns.boxplot(x=df['amount'])
plt.title("Transaction Amount Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Drop unnecessary columns
df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Encode transaction type
df = pd.get_dummies(df, columns=['type'], drop_first=True)

# Define features and target
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=500, class_weight='balanced', solver='liblinear')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate model
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))

# Feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\nTop Features:\n", importance.head(10))


#Q2. Logistic Regression use kiya kyunki
# ye binary fraud detection problem hai. 
#Ye har transaction ke liye probability nikalta hai aur us basis par decision leta hai.

#Q3. Name columns hata diye kyunki wo useful nahi the. 
#Correlation check karke sirf relevant features ko model me rakha.

#Q4. Model ko confusion matrix, recall aur ROC-AUC se check kiya. 
#Recall high aaya matlab zyada fraud cases detect ho gaye.

#Q5. TRANSFER aur CASH_OUT type transactions fraud predict karne me important rahe. 
#Amount aur balance changes ka bhi effect dikha.

#Q6. Ye factors practical lagte hain kyunki fraud me paise transfer karke cash out kiye jaate hain. 
#Isliye model ka result real situation se match karta hai.

#Q7. Company ko high amount transfers par monitoring aur extra verification lagani chahiye. 
#Suspicious accounts ko temporarily block bhi kiya ja sakta hai.

#Q8. Fraud rate aur financial loss ko time ke saath compare karke result measure kiya ja sakta hai. 
#False positives bhi monitor karne chahiye.