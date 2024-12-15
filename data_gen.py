import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_banking_customer_data(num_customers=1000, start_date='2020-01-01', end_date='2024-12-31'):
    """
    Generate synthetic banking customer data for CLV analysis
    
    Parameters:
    num_customers (int): Number of customers to generate
    start_date (str): Start date for transaction history
    end_date (str): End date for transaction history
    
    Returns:
    tuple: (customers_df, transactions_df, products_df)
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate customer base data
    customers = []
    age_ranges = [(25, 35), (35, 50), (50, 65), (65, 80)]
    income_ranges = [(30000, 60000), (60000, 100000), (100000, 250000), (250000, 500000)]
    
    for customer_id in range(num_customers):
        age_range = np.random.choice(len(age_ranges), p=[0.3, 0.4, 0.2, 0.1])
        age = np.random.randint(age_ranges[age_range][0], age_ranges[age_range][1])
        
        customers.append({
            'customer_id': customer_id,
            'age': age,
            'income': np.random.randint(income_ranges[age_range][0], income_ranges[age_range][1]),
            'credit_score': np.random.normal(700, 50),
            'tenure_months': np.random.randint(1, 120),
            'region': np.random.choice(['North', 'South', 'East', 'West']),
            'acquisition_channel': np.random.choice(['Online', 'Branch', 'Referral', 'Marketing'], p=[0.4, 0.3, 0.2, 0.1])
        })
    
    customers_df = pd.DataFrame(customers)
    
    # Generate product holdings
    products = []
    product_types = ['Savings', 'Checking', 'Credit Card', 'Mortgage', 'Investment', 'Personal Loan']
    
    for customer_id in range(num_customers):
        num_products = np.random.randint(1, 4)
        customer_products = np.random.choice(product_types, num_products, replace=False)
        
        for product in customer_products:
            products.append({
                'customer_id': customer_id,
                'product_type': product,
                'start_date': pd.Timestamp(start_date) + pd.Timedelta(days=np.random.randint(0, 365)),
                'balance': np.random.lognormal(10, 1) if product != 'Credit Card' else 0,
                'status': 'Active'
            })
    
    products_df = pd.DataFrame(products)
    
    # Generate transactions
    transactions = []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    
    transaction_types = {
        'Deposit': (500, 5000, 1),
        'Withdrawal': (-3000, -100, 1),
        'Transfer': (-1000, 1000, 0.8),
        'Bill Payment': (-500, -50, 0.9),
        'Fee': (-50, -10, 0.3)
    }
    
    for customer_id in range(num_customers):
        # Number of transactions varies by customer
        num_transactions = np.random.randint(50, 200)
        
        for _ in range(num_transactions):
            trans_type = np.random.choice(list(transaction_types.keys()))
            min_amount, max_amount, _ = transaction_types[trans_type]
            
            transactions.append({
                'customer_id': customer_id,
                'transaction_date': start_ts + (end_ts - start_ts) * np.random.random(),
                'transaction_type': trans_type,
                'amount': np.random.uniform(min_amount, max_amount),
                'channel': np.random.choice(['Online', 'Mobile', 'Branch', 'ATM'], p=[0.5, 0.3, 0.1, 0.1])
            })
    
    transactions_df = pd.DataFrame(transactions)
    transactions_df = transactions_df.sort_values('transaction_date')
    
    # Clean and format data
    customers_df['credit_score'] = customers_df['credit_score'].clip(300, 850).round()
    products_df['balance'] = products_df['balance'].round(2)
    transactions_df['amount'] = transactions_df['amount'].round(2)
    
    return customers_df, transactions_df, products_df

def calculate_basic_metrics(customers_df, transactions_df, products_df):
    """
    Calculate basic customer metrics for CLV analysis
    """
    metrics = []
    
    for customer_id in customers_df['customer_id']:
        customer_transactions = transactions_df[transactions_df['customer_id'] == customer_id]
        customer_products = products_df[products_df['customer_id'] == customer_id]
        
        metrics.append({
            'customer_id': customer_id,
            'total_transaction_amount': customer_transactions['amount'].sum(),
            'transaction_count': len(customer_transactions),
            'product_count': len(customer_products),
            'avg_transaction_value': customer_transactions['amount'].mean(),
            'total_balance': customer_products['balance'].sum()
        })
    
    return pd.DataFrame(metrics)

def save_data_to_csv(customers_df, transactions_df, products_df, metrics_df, output_dir='data'):
    """
    Save all dataframes to CSV files in the specified directory
    
    Parameters:
    customers_df, transactions_df, products_df, metrics_df: DataFrame objects
    output_dir (str): Directory to save the CSV files
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Save each DataFrame to CSV
    customers_df.to_csv(f"{output_dir}/customers.csv", index=False)
    transactions_df.to_csv(f"{output_dir}/transactions.csv", index=False)
    products_df.to_csv(f"{output_dir}/products.csv", index=False)
    metrics_df.to_csv(f"{output_dir}/customer_metrics.csv", index=False)
    
    print(f"Files saved successfully in {output_dir}:")
    print(f"- customers.csv: {len(customers_df)} records")
    print(f"- transactions.csv: {len(transactions_df)} records")
    print(f"- products.csv: {len(products_df)} records")
    print(f"- customer_metrics.csv: {len(metrics_df)} records")

# Generate and save the data
if __name__ == "__main__":
    # Generate the data
    customers_df, transactions_df, products_df = generate_banking_customer_data()
    metrics_df = calculate_basic_metrics(customers_df, transactions_df, products_df)
    
    # Save to CSV files
    save_data_to_csv(customers_df, transactions_df, products_df, metrics_df)