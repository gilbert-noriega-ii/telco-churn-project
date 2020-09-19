from env import host, user, password
import seaborn as sns
import pandas as pd
import numpy as np
import os



def get_connection(db, user = user, host = host, password = password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


def new_telco_data():
    sql_query = """
                SELECT * 
                FROM customers 
                JOIN contract_types using (contract_type_id)
                JOIN internet_service_types using (internet_service_type_id)
                JOIN payment_types using (payment_type_id);
                """
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    df.to_csv('telco.csv')
    return df


def get_telco_data(cached=False):
    if cached or os.path.isfile('telco.csv') == False:
        df = new_telco_data()
    else:
        df = pd.read_csv('telco.csv', index_col=0)
    return df