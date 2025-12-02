import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('train.csv')

# Clean NA
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df.dropna(inplace=True)

# Encode
le = LabelEncoder()
cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
for col in cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Scale
scaler = StandardScaler()
X[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']] = \
    scaler.fit_transform(X[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']])

# Train
model = LogisticRegression()
model.fit(X, y)

# Save
pickle.dump(model, open('loan_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
