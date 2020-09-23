import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from acquire import get_telco_data

def online_checker(row):
        if row == 'Yes':
            return 1
        else:
            return 0


def telco_split(df):

    train_validate, test = train_test_split(df, test_size=.15, 
                                        random_state=123, 
                                        stratify=df.churn_Yes)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.churn_Yes)
    return train, validate, test


def prep_telco_data(cached = True):
    # use my acquire function to read data into a df from a csv file
    df = pd.DataFrame(get_telco_data())
    #set index as customer_id
    df = df.set_index('customer_id')
    #rows full of Nan values
    rows_to_drop = ([6895, 6686, 6615, 6605, 6569, 6143, 2132, 2048, 2029, 1949, 1878])
    #dropping the Nan values
    df.drop(df.index[rows_to_drop], inplace = True)
    #changing data type of total charges to float
    df.total_charges = pd.to_numeric(df.total_charges)
    # change device_protection into numeric values
    df.device_protection = df.device_protection.apply(online_checker)
    # change tech_suppport into numeric values
    df.tech_support = df.tech_support.apply(online_checker)
    # change online security into numeric values
    df.online_security = df.online_security.apply(online_checker)
    #change online backup into numeric values
    df.online_backup = df.online_backup.apply(online_checker)
    #change streaming tv into numeric values
    df.streaming_tv = df.streaming_tv.apply(online_checker)
    #change streaming movies into numeric values
    df.streaming_movies = df.streaming_movies.apply(online_checker)
    #change multiple lines into numeric values
    df.multiple_lines = df.multiple_lines.apply(online_checker)
    #create a new column with tenure by the year
    df['tenure_by_year'] = df.tenure / 12
    #change multiple_lines into a numeric value
    df.multiple_lines = df.multiple_lines.apply(online_checker)
    #create dummy values for gender, churn, and paperless_billing
    telco_dummies = pd.get_dummies(df[['gender', 'partner', 'dependents', 'phone_service', 'churn', 'paperless_billing']], drop_first=True)
    #create dummy values for internet
    internet_dummie = pd.get_dummies(df['internet_service_type_id'])
    #rename the dummy columns
    internet_dummie = internet_dummie.rename(columns = {1:'DSL', 2:'Fiber', 3:'No_internet'})
    #create dummy variable for payment
    payment_dummie = pd.get_dummies(df['payment_type_id'])
    #rename the dummy columns
    payment_dummie = payment_dummie.rename(columns = {1: 'echeck', 2: 'mailed_check', 3: 'bank_transfer', 4:'credit_card'})
    #create dummy variable for contract type
    contract_dummie = pd.get_dummies(df['contract_type_id'])
    #rename dummy columns
    contract_dummie = contract_dummie.rename(columns = {1:'Month2month', 2:'1year', 3:'2year'})
    #add dummy values into the main dataframe
    df = pd.concat([df, telco_dummies, payment_dummie, contract_dummie, internet_dummie], axis=1)
    #list duplicate columns
    col_to_drop = ['payment_type_id', 'gender', 'partner', 'dependents', 'phone_service', 'churn', 'paperless_billing', 'contract_type', 'internet_service_type', 'payment_type']
    #drop duplicate columns
    df = df.drop(columns = col_to_drop)
    #split data into train, validate and test subsets
    train, validate, test = telco_split(df)
    return train, validate, test
