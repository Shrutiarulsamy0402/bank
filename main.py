import streamlit as st
import pandas as pd
import os
import random
import numpy as np
from xgboost import XGBClassifier

# Paths to CSV files
data_path = "data"
users_file = os.path.join(data_path, "users.csv")
accounts_file = os.path.join(data_path, "accounts.csv")
loans_file = os.path.join(data_path, "loan_applications.csv")
loan_status_file = os.path.join(data_path, "loan_status.csv")
transactions_file = os.path.join(data_path, "transactions.csv")

# Load and Save CSV
def load_csv(file, expected_columns=None):
    try:
        if os.path.exists(file):
            df = pd.read_csv(file)
            if expected_columns:
                for col in expected_columns:
                    if col not in df.columns:
                        df[col] = np.nan
            return df
        else:
            return pd.DataFrame(columns=expected_columns if expected_columns else [])
    except Exception as e:
        st.error(f"Error loading {file}: {e}")
        return pd.DataFrame(columns=expected_columns if expected_columns else [])

def save_csv(df, file):
    try:
        df.to_csv(file, index=False)
    except Exception as e:
        st.error(f"Error saving {file}: {e}")

def load_data_to_session():
    st.session_state.users_df = load_csv(users_file, ["user_id", "username", "password", "role"])
    st.session_state.accounts_df = load_csv(accounts_file, ["user_id", "account_no", "address", "mobile", "balance"])
    st.session_state.loans_df = load_csv(loans_file, ["loan_id", "user_id", "amount", "purpose", "income", "status", "application_date", "remarks"])
    st.session_state.loan_status_df = load_csv(loan_status_file, ["loan_id", "user_id", "amount", "purpose", "income", "status", "application_date", "remarks"])
    st.session_state.transactions_df = load_csv(transactions_file, ["user_id", "loan_id", "amount", "method", "date"])

load_data_to_session()

users_df = st.session_state.users_df
accounts_df = st.session_state.accounts_df
loans_df = st.session_state.loans_df
loan_status_df = st.session_state.loan_status_df
transactions_df = st.session_state.transactions_df

if "user" not in st.session_state:
    st.session_state.user = None

def ensure_columns():
    global users_df, accounts_df, loans_df
    if 'status' not in loans_df.columns:
        loans_df['status'] = 'pending'
    if 'remarks' not in loans_df.columns:
        loans_df['remarks'] = ''
    for col in ['account_no', 'address', 'balance', 'mobile']:
        if col not in accounts_df.columns:
            accounts_df[col] = '' if col != 'balance' else 0
    for col in ['username', 'user_id', 'password', 'role']:
        if col not in users_df.columns:
            users_df[col] = ''

ensure_columns()
# User Registration
def create_new_user():
    st.title("Create New User Account")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    role = st.selectbox("Role", ["user"])
    city = st.text_input("City")
    mobile = st.text_input("Mobile Number (e.g., xxxxxxx237)")

    if st.button("Create Account"):
        if username in users_df["username"].values:
            st.error("Username already exists. Please choose another.")
        else:
            user_id = f"U{len(users_df)+1:04d}"
            new_user = pd.DataFrame([{"user_id": user_id, "username": username, "password": password, "role": role}])
            new_account = pd.DataFrame([{"user_id": user_id, "account_no": f"XXXXXXX{random.randint(100,999)}", "address": city, "mobile": mobile, "balance": 0}])
            save_csv(pd.concat([users_df, new_user], ignore_index=True), users_file)
            save_csv(pd.concat([accounts_df, new_account], ignore_index=True), accounts_file)
            st.success("Account created successfully!")

def login():
    st.title("Indian Bank")
    menu = st.radio("Select an option", ["Login", "Create Account", "Forgot Password?"])

    if menu == "Create Account":
        create_new_user()
        return

    if menu == "Forgot Password?":
        st.subheader("Reset Your Password with Mobile Verification")
        username = st.text_input("Enter your username")
        mobile = st.text_input("Enter your registered mobile number")
        new_password = st.text_input("Enter your new password", type="password")

        if st.button("Reset Password"):
            users_df = load_csv(users_file)
            accounts_df = load_csv(accounts_file)
            user_row = users_df[users_df["username"] == username]
            if user_row.empty:
                st.error("❌ Username not found.")
                return
            user_id = user_row.iloc[0]["user_id"]
            acc_row = accounts_df[(accounts_df["user_id"] == user_id) & (accounts_df["mobile"] == mobile)]
            if acc_row.empty:
                st.error("❌ Mobile number does not match our records.")
            else:
                users_df.loc[users_df["username"] == username, "password"] = new_password
                save_csv(users_df, users_file)
                st.success("✅ Password reset successful!")
        return

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        users_df = load_csv(users_file)
        required_cols = {"username", "password", "role", "user_id"}
        if not required_cols.issubset(set(users_df.columns)):
            st.error("Error: 'users.csv' is missing required columns.")
            st.stop()
        user = users_df[(users_df["username"] == username) & (users_df["password"] == password)]
        if not user.empty:
            st.session_state.user = user.iloc[0].to_dict()
            st.success(f"Logged in as {username}")
            st.rerun()
        else:
            st.error("Invalid username or password")

# MAIN LOGIC
if st.session_state.user:
    st.sidebar.write(f"👋 Welcome, {st.session_state.user['username']}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()
else:
    login()
# Admin Dashboard
def admin_dashboard():
    import matplotlib.pyplot as plt
    st.sidebar.title("Admin Panel")
    option = st.sidebar.radio("Select", [
        "📃 All Applications",
        "✅ Pending Loans",
        "📊 Loan Summary"
    ])

    if option == "📃 All Applications":
        st.subheader("All Loan Applications")
        sort_option = st.selectbox("Filter by Status", ["All", "approved", "pending", "declined"])
        display_df = loans_df if sort_option == "All" else loans_df[loans_df["status"] == sort_option]
        st.dataframe(display_df.sort_values("application_date", ascending=False))

    elif option == "✅ Pending Loans":
        st.subheader("Pending Loans (Average Risk)")
        pending_loans = loans_df[loans_df["status"] == "pending"]
        if pending_loans.empty:
            st.info("No pending loans to review.")
        else:
            for idx, row in pending_loans.iterrows():
                loan_id = row["loan_id"]
                st.write(f"### Loan ID: {loan_id}")
                st.write(row)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"✅ Approve {loan_id}"):
                        loans_df.loc[loans_df["loan_id"] == loan_id, "status"] = "approved"
                        loans_df.loc[loans_df["loan_id"] == loan_id, "remarks"] = "Admin-approved"
                        save_csv(loans_df, loans_file)
                        save_csv(loans_df, loan_status_file)
                        st.success(f"Loan {loan_id} approved")
                        st.rerun()
                with col2:
                    if st.button(f"❌ Decline {loan_id}"):
                        reason = random.choice([
                            "Low credit score",
                            "Debt to income ratio too high",
                            "Insufficient documentation"
                        ])
                        loans_df.loc[loans_df["loan_id"] == loan_id, "status"] = "declined"
                        loans_df.loc[loans_df["loan_id"] == loan_id, "remarks"] = f"Admin-declined: {reason}"
                        save_csv(loans_df, loans_file)
                        save_csv(loans_df, loan_status_file)
                        st.error(f"Loan {loan_id} declined: {reason}")
                        st.rerun()

    elif option == "📊 Loan Summary":
        st.subheader("Loan Summary")
        st.metric("Total", len(loans_df))
        st.metric("Approved", len(loans_df[loans_df["status"] == "approved"]))
        st.metric("Declined", len(loans_df[loans_df["status"] == "declined"]))
        st.metric("Pending", len(loans_df[loans_df["status"] == "pending"]))

# User Dashboard
def user_dashboard():
    st.sidebar.title("User Dashboard")
    choice = st.sidebar.radio("Menu", [
        "📈 Account Summary", "📝 Apply for Loan", "📊 Loan Status"
    ])

    user_id = st.session_state.user["user_id"]

    if choice == "📈 Account Summary":
        st.subheader("Your Account Summary")
        acc = accounts_df[accounts_df["user_id"] == user_id]
        st.dataframe(acc)

    elif choice == "📝 Apply for Loan":
        st.subheader("Loan Application")
        amount = st.number_input("Loan Amount", min_value=1000)
        purpose = st.selectbox("Purpose", ["Education", "Medical", "Business", "Personal"])
        income = st.number_input("Monthly Income", min_value=0)

        if st.button("Submit Application"):
            loan_id = f"L{len(loans_df)+1:03d}"

            train_df = loans_df[loans_df["status"] != "pending"]
            if not train_df.empty and len(train_df["status"].unique()) > 1:
                train_df = train_df[["amount", "income", "status"]].dropna()
                X_train = train_df[["amount", "income"]]
                y_train = (train_df["status"] == "approved").astype(int)
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(X_train, y_train)
                prob = model.predict_proba([[amount, income]])[0][1]
                risk_score = round((1 - prob) * 100, 2)
                if risk_score <= 39:
                    status = "approved"
                    remarks = f"Auto-approved. Risk Score: {risk_score}%"
                elif risk_score >= 61:
                    reason = random.choice([
                        "Low credit score", "High debt-to-income ratio",
                        "Insufficient income", "Incomplete documentation"
                    ])
                    status = "declined"
                    remarks = f"Auto-declined: {reason}. Risk Score: {risk_score}%"
                else:
                    status = "pending"
                    remarks = f"Pending admin review. Risk Score: {risk_score}%"
            else:
                status = "pending"
                remarks = "Pending admin review (not enough training data)"

            new_loan = {
                "loan_id": loan_id, "user_id": user_id, "amount": amount,
                "purpose": purpose, "income": income, "status": status,
                "application_date": pd.Timestamp.today().strftime('%Y-%m-%d'),
                "remarks": remarks
            }

            loans_df_updated = pd.concat([loans_df, pd.DataFrame([new_loan])], ignore_index=True)
            save_csv(loans_df_updated, loans_file)
            save_csv(loans_df_updated, loan_status_file)
            st.success(f"Loan submitted! Current status: **{status.upper()}**")

    elif choice == "📊 Loan Status":
        st.subheader("Your Loan Applications")
        st.dataframe(loans_df[loans_df["user_id"] == user_id])

# Launch correct dashboard
if st.session_state.user:
    if st.session_state.user["role"] == "admin":
        admin_dashboard()
    else:
        user_dashboard()
