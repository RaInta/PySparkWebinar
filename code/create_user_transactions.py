#!/usr/bin/env python
#
###########################################
#
# File: create_user_transactions.py
# Author: Ra Inta
# Description:
# Created: April 04, 2019
# Last Modified: 20210330, R.I.
#
###########################################

from mimesis import Person, Address, Business, Payment, Text

from scipy.stats import pareto
import pandas as pd
import numpy as np

from pathlib import Path

# Note: we don't ever store user passwords as clear text!!!
# To emulate salting and hashing the user passwords:
import hashlib, uuid

# However, we should really use a dedicated password hashing
# package, such as passlib. However, this is out of scope
# for this demo (it would require further installation).
# import passlib

import datetime as dt

DATA_DIR = Path("D:\Shared\Webinars\PySpark\data")

TRANSACTIONS_DIR = DATA_DIR / "transactions"

USERS_DIR = DATA_DIR / "users"

TRANSACTIONS_DIR.mkdir(exist_ok=True)
USERS_DIR.mkdir(exist_ok=True)

np.random.seed(42)  # To make our analysis reproducible

person = Person()
address = Address()
business = Business()
payment = Payment()
text = Text()

##################################################
### Define a couple of convenience functions:
##################################################

def hashed_passwd(passwd):
    """We should never entertain the idea of storing users' passwords
    as plaintext. This function performs a basic salting and hashing
    of a password input. This function should *never* be used in a
    production setting; if you need to securely store salted and hashed
    passwords, use a dedicated package such as passlib."""
    salt = uuid.uuid4().hex
    return hashlib.sha512(passwd.encode('utf-8')
                          + salt.encode('utf-8')).hexdigest()


def account_balance():
    """Generate account balances according to a Pareto distribution.
    We should expect balances to be distributed as with other income
    distributions.  The power exponent is chosen here to replicate
    the 80-20 rule."""
    return float(pareto.rvs(1.161))

def generate_transaction_amount(account_balance):
    """Generate transaction amount, proportional to the users' account
    balance.
    This is normally distributed with a std dev of 1,
    and truncated with a minimum absolute transaction of 0.01
    and a maximum of 30% of the users' current balance"""
    mean_fraction = 0.05  # Average transaction, percentage of balance
    min_transaction = 0.01
    max_fraction = 0.3
    rnd_amplitude = np.random.randn() + mean_fraction
    # Clip to maximum fraction of balance
    if rnd_amplitude > max_fraction:
        rnd_amplitude = max_fraction
    transaction_amount = rnd_amplitude*account_balance
    # Enforce minimum absolute transaction
    if transaction_amount < min_transaction:
        transaction_amount = min_transaction
    return transaction_amount

def generate_transaction_date(n_samples,
                              max_date=dt.datetime.today(), 
                              min_date=dt.datetime(2019, 1, 1, 12, 0)):
    rnd_date = [min_date + dt.timedelta(seconds=x*(max_date-min_date).total_seconds())
                for x in np.random.rand(n_samples)]
    return rnd_date


##################################################

##################################################
### Generate a DataFrame of user information
##################################################
# Generate e.g. 200,000 rows of the following:
# user_id, first_name, last_name, email, password, employer, address,
# birth_date, credit_card_num, credit_card_exp, security_answer,
# account_balance


#csv_name = "/home/ra/host/BH_Analytics/spark/courseware/data/gapminder.csv"
# df = sqlContext.read.csv("/home/ra/host/BH_Analytics/spark/courseware/data/gapminder.csv", header=True)

user_columns = ["user_id", "first_name", "last_name", "email",
        "password_hashed", "address", "city", "state", "employer", "age",
        "credit_card_num", "credit_card_exp",
        "security_answer", "account_balance" ]

data_length = 200_000

user_df = pd.DataFrame([[x, person.name(), person.surname(), person.email(),
            hashed_passwd(person.password()), address.address(),
            address.city(), address.state(),
            business.company(), person.age(), payment.credit_card_number(),
            payment.credit_card_expiration_date(), text.word(),
                         account_balance()] for x in range(data_length)], 
                                columns=user_columns)

user_df["gender"] = np.where(np.random.randint(0, 1, data_length), 
                             "male", "female")

user_df.loc[user_df.sample(frac=0.6*np.random.rand(), random_state=42).index, 
            "age"] = np.NaN
user_df.loc[user_df.sample(frac=0.6*np.random.rand(), random_state=42).index, 
            "gender"] = np.NaN
user_df.loc[user_df.sample(frac=0.6*np.random.rand(), random_state=42).index, 
            "address"] = np.NaN
user_df.loc[user_df.sample(frac=0.6*np.random.rand(), random_state=42).index, 
            "employer"] = np.NaN
user_df.loc[user_df.sample(frac=0.6*np.random.rand(), random_state=42).index, 
            "security_answer"] = np.NaN
user_df.loc[user_df.sample(frac=0.6*np.random.rand(), random_state=42).index, 
            "last_name"] = np.NaN

print("Amount of missingness in user data:")
print(user_df.isnull().mean())

transaction_df = user_df[["user_id", "account_balance"]]\
    .sample(frac=1.0, replace=True, random_state=42)

transaction_df["transaction_amount"] = transaction_df["account_balance"]\
    .apply(generate_transaction_amount)

transaction_df["transaction_date"] = generate_transaction_date(data_length)


transaction_df["year_month"] = transaction_df["transaction_date"]\
    .dt.strftime("%Y_%b")

print("Summary of user data:")
print(user_df.drop(columns=["user_id"]).describe())

print("Sample of user data:")
print(user_df.head())

print("Summary of transaction data:")
print(transaction_df.drop(columns=["user_id"]).describe())

print("Sample of transaction data:")
print(transaction_df.head())

# Output transaction files
for year_month, data in transaction_df.groupby("year_month"):
    TRANSACTION_FILE = TRANSACTIONS_DIR / f"transactions_{year_month}.csv"
    data.to_csv(TRANSACTION_FILE, index=False)

# Output user file
USER_FILE = USERS_DIR / "users.csv"
user_df.to_csv(USER_FILE)

##################################################

##################################################
### Perform some Exploratory Data Analysis
##################################################

# user_df.sample(5)

# user_df.describe()

# Note the median balance is 1.8, while the mean is 5.3
# Recall we generated a heavily skewed distribution!

# I designed it according to the famous "80-20 rule"
# The top twenty percent own 80% of the balances.
# Let's test it. Take the 80th percentile:
# critical80 = np.quantile(user_df["account_balance"], 0.8)
## 4.013269256450965

# the_few = user_df.loc[user_df["account_balance"] > critical80,
#                       "account_balance"].sum()

# tot_balance = user_df["account_balance"].sum()

# the_few/tot_balance
## 0.7298469832819879
# So here, the top 20% 'only' have 73% of the account balance

# Some limitations of mimesis
# If you want realistic distributions of certain numerical variables
# then you should simulate populations yourself. E.g.:

# user_df["age"].plot(kind="kde")

# The way ages are generated are not exactly samples of any real population!
# This will depend on the underlying demographic dynamics.


##################################################


##################################################
### Export data to Excel and print summary
##################################################

# print("Account balance for top 20% of users: {} \nFraction of total \
#       balance owned by top 20%: {}%\n".format(critical80,
#                                               100*the_few/tot_balance))

# output_file_name = "synthetic_user_data.xlsx"

# user_df.to_excel(output_file_name, index=False)

# print("Synthetic user data output to: {}\n".format(output_file_name))

##################################################


###########################################
# End of create_user_transactions.py
###########################################
