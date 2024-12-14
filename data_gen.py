import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import random
from typing import Dict, List
import json
from pathlib import Path

class EnterpriseBankingDataGenerator:
    def __init__(self, start_date="2023-01-01", num_customers=1000, num_months=24):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.num_customers = num_customers
        self.num_months = num_months
        self.date_range = pd.date_range(start=self.start_date, 
                                      periods=num_months * 30, 
                                      freq='D')
        
        # Create data directory if it doesn't exist
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize static data
        self.branch_data = self._generate_branch_data()
        self.product_catalog = self._generate_product_catalog()
        self.merchant_data = self._generate_merchant_data()
        
    def _generate_branch_data(self) -> pd.DataFrame:
        """Generate bank branch information"""
        states = ['CA', 'NY', 'TX', 'FL', 'IL', 'MA', 'WA', 'CO', 'GA', 'VA']
        branches = []
        branch_id = 1000
        
        for state in states:
            num_branches = np.random.randint(5, 15)
            for _ in range(num_branches):
                branch_id += 1
                branches.append({
                    'branch_id': branch_id,
                    'branch_name': f'Branch {branch_id}',
                    'state': state,
                    'zip_code': np.random.randint(10000, 99999),
                    'branch_type': np.random.choice(['Full Service', 'Limited Service', 'ATM Only']),
                    'opening_date': self.start_date - timedelta(days=np.random.randint(365, 3650)),
                    'employee_count': np.random.randint(5, 50),
                    'atm_count': np.random.randint(1, 5),
                    'safe_deposit_boxes': np.random.choice([True, False], p=[0.7, 0.3]),
                    'drive_through': np.random.choice([True, False], p=[0.6, 0.4])
                })
        
        return pd.DataFrame(branches)

    def _generate_product_catalog(self) -> pd.DataFrame:
        """Generate banking product catalog"""
        products = [
            # Checking Accounts
            {'product_id': 'CHK_BASIC', 'product_name': 'Basic Checking', 'product_type': 'CHECKING',
             'monthly_fee': 0, 'min_balance': 0, 'interest_rate': 0.0},
            {'product_id': 'CHK_PLUS', 'product_name': 'Premium Checking', 'product_type': 'CHECKING',
             'monthly_fee': 12, 'min_balance': 1500, 'interest_rate': 0.01},
            {'product_id': 'CHK_PREMIER', 'product_name': 'Premier Checking', 'product_type': 'CHECKING',
             'monthly_fee': 25, 'min_balance': 5000, 'interest_rate': 0.02},
            
            # Savings Accounts
            {'product_id': 'SAV_BASIC', 'product_name': 'Basic Savings', 'product_type': 'SAVINGS',
             'monthly_fee': 0, 'min_balance': 300, 'interest_rate': 0.02},
            {'product_id': 'SAV_PLUS', 'product_name': 'Premium Savings', 'product_type': 'SAVINGS',
             'monthly_fee': 5, 'min_balance': 2500, 'interest_rate': 0.03},
            {'product_id': 'SAV_MONEY_MARKET', 'product_name': 'Money Market', 'product_type': 'SAVINGS',
             'monthly_fee': 10, 'min_balance': 10000, 'interest_rate': 0.04},
            
            # Credit Cards
            {'product_id': 'CC_BASIC', 'product_name': 'Basic Credit Card', 'product_type': 'CREDIT_CARD',
             'annual_fee': 0, 'apr': 0.1499, 'rewards_rate': 0.01},
            {'product_id': 'CC_REWARDS', 'product_name': 'Rewards Credit Card', 'product_type': 'CREDIT_CARD',
             'annual_fee': 95, 'apr': 0.1699, 'rewards_rate': 0.015},
            {'product_id': 'CC_PREMIUM', 'product_name': 'Premium Credit Card', 'product_type': 'CREDIT_CARD',
             'annual_fee': 495, 'apr': 0.1899, 'rewards_rate': 0.025},
            
            # Loans
            {'product_id': 'LOAN_AUTO', 'product_name': 'Auto Loan', 'product_type': 'LOAN',
             'min_amount': 5000, 'max_amount': 50000, 'base_rate': 0.0399},
            {'product_id': 'LOAN_PERSONAL', 'product_name': 'Personal Loan', 'product_type': 'LOAN',
             'min_amount': 1000, 'max_amount': 25000, 'base_rate': 0.0699},
            {'product_id': 'LOAN_HOME', 'product_name': 'Home Loan', 'product_type': 'LOAN',
             'min_amount': 100000, 'max_amount': 1000000, 'base_rate': 0.0299}
        ]
        
        return pd.DataFrame(products)

    def _generate_merchant_data(self) -> pd.DataFrame:
        """Generate merchant profiles with detailed attributes"""
        categories = {
            'GROCERY': ['Walmart', 'Whole Foods', 'Trader Joes', 'Safeway', 'Kroger'],
            'RETAIL': ['Amazon', 'Target', 'Best Buy', 'Macys', 'Nike'],
            'ENTERTAINMENT': ['Netflix', 'AMC Theaters', 'Spotify', 'Steam', 'Disney+'],
            'UTILITY': ['AT&T', 'PG&E', 'Comcast', 'Verizon', 'Water Corp'],
            'HEALTHCARE': ['CVS', 'Walgreens', 'Kaiser', 'United Health', 'Blue Cross'],
            'TRAVEL': ['United Airlines', 'Marriott', 'Expedia', 'Airbnb', 'Enterprise'],
            'DINING': ['McDonalds', 'Starbucks', 'Chipotle', 'Subway', 'Local Cafe'],
            'FINANCIAL': ['Venmo', 'PayPal', 'Zelle', 'Cash App', 'Wire Transfer']
        }
        
        merchants = []
        merchant_id = 1000
        
        for category, names in categories.items():
            for name in names:
                merchant_id += 1
                merchants.append({
                    'merchant_id': merchant_id,
                    'merchant_name': name,
                    'category': category,
                    'avg_transaction': np.random.lognormal(4, 0.5),
                    'weekend_multiplier': np.random.uniform(1.1, 1.5) if category in ['ENTERTAINMENT', 'DINING', 'RETAIL'] else 1.0,
                    'seasonal_pattern': np.random.choice(['NONE', 'SUMMER', 'WINTER', 'HOLIDAY']),
                    'mcc_code': np.random.randint(1000, 9999),
                    'online_presence': np.random.choice([True, False], p=[0.7, 0.3]),
                    'international': np.random.choice([True, False], p=[0.2, 0.8]),
                    'high_risk': np.random.choice([True, False], p=[0.1, 0.9])
                })
        
        return pd.DataFrame(merchants)

    def generate_customers(self) -> pd.DataFrame:
        """Generate core customer information"""
        print("Generating customer data...")
        
        customers = []
        for cust_id in range(1, self.num_customers + 1):
            birth_date = self.start_date - timedelta(days=np.random.randint(365*18, 365*80))
            onboarding_date = self.start_date - timedelta(days=np.random.randint(0, 365*10))
            
            customers.append({
                'customer_id': cust_id,
                'ssn': f"{np.random.randint(100, 999)}-{np.random.randint(10, 99)}-{np.random.randint(1000, 9999)}",
                'first_name': f"FirstName_{cust_id}",
                'last_name': f"LastName_{cust_id}",
                'birth_date': birth_date,
                'email': f"customer_{cust_id}@email.com",
                'phone_number': f"{np.random.randint(100, 999)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}",
                'address_street': f"{np.random.randint(100, 9999)} Main St",
                'address_city': 'SomeCity',
                'address_state': np.random.choice(self.branch_data['state'].unique()),
                'address_zip': np.random.randint(10000, 99999),
                'onboarding_date': onboarding_date,
                'onboarding_branch_id': np.random.choice(self.branch_data['branch_id']),
                'credit_score': np.random.normal(700, 80).clip(300, 850),
                'income': np.random.lognormal(11, 0.5).clip(20000, 300000),
                'occupation': np.random.choice(['Professional', 'Service', 'Technical', 'Student', 'Retired']),
                'employer': f"Employer_{np.random.randint(1, 100)}",
                'customer_segment': np.random.choice(['MASS', 'AFFLUENT', 'PRIVATE', 'BUSINESS'], p=[0.7, 0.2, 0.05, 0.05])
            })
        
        return pd.DataFrame(customers)

    def generate_accounts(self, customers: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate different types of account information"""
        print("Generating account data...")
        
        checking_accounts = []
        savings_accounts = []
        credit_cards = []
        loans = []
        
        for _, customer in customers.iterrows():
            # Checking accounts
            num_checking = np.random.choice([1, 2], p=[0.8, 0.2])
            for _ in range(num_checking):
                product = np.random.choice(self.product_catalog[self.product_catalog['product_type'] == 'CHECKING']['product_id'])
                checking_accounts.append({
                    'account_id': f"CHK_{len(checking_accounts) + 1000}",
                    'customer_id': customer['customer_id'],
                    'product_id': product,
                    'open_date': customer['onboarding_date'],
                    'status': 'ACTIVE',
                    'current_balance': np.random.lognormal(8, 1),
                    'avg_balance_3m': np.random.lognormal(8, 1),
                    'avg_balance_6m': np.random.lognormal(8, 1),
                    'avg_balance_12m': np.random.lognormal(8, 1),
                    'overdraft_count': np.random.poisson(0.5)
                })
            
            # Savings accounts
            if np.random.random() < 0.8:  # 80% have savings
                product = np.random.choice(self.product_catalog[self.product_catalog['product_type'] == 'SAVINGS']['product_id'])
                savings_accounts.append({
                    'account_id': f"SAV_{len(savings_accounts) + 1000}",
                    'customer_id': customer['customer_id'],
                    'product_id': product,
                    'open_date': customer['onboarding_date'] + timedelta(days=np.random.randint(0, 365)),
                    'status': 'ACTIVE',
                    'current_balance': np.random.lognormal(9, 1),
                    'avg_balance_3m': np.random.lognormal(9, 1),
                    'avg_balance_6m': np.random.lognormal(9, 1),
                    'avg_balance_12m': np.random.lognormal(9, 1),
                    'withdrawal_count_6m': np.random.poisson(2)
                })
            
            # Credit cards
            if np.random.random() < 0.7:  # 70% have credit cards
                num_cards = np.random.choice([1, 2], p=[0.8, 0.2])
                for _ in range(num_cards):
                    product = np.random.choice(self.product_catalog[self.product_catalog['product_type'] == 'CREDIT_CARD']['product_id'])
                    credit_cards.append({
                        'account_id': f"CC_{len(credit_cards) + 1000}",
                        'customer_id': customer['customer_id'],
                        'product_id': product,
                        'open_date': customer['onboarding_date'] + timedelta(days=np.random.randint(0, 365)),
                        'status': 'ACTIVE',
                        'credit_limit': np.random.lognormal(9, 0.5),
                        'current_balance': np.random.lognormal(8, 1),
                        'last_payment_amount': np.random.lognormal(6, 1),
                        'last_payment_date': self.start_date - timedelta(days=np.random.randint(1, 30)),
                        'missed_payment_count': np.random.poisson(0.2),
                        'utilization_rate': np.random.beta(2, 5)
                    })
            
            # Loans
            if np.random.random() < 0.4:  # 40% have loans
                product = np.random.choice(self.product_catalog[self.product_catalog['product_type'] == 'LOAN']['product_id'])
                loan_amount = np.random.lognormal(11, 1)
                loans.append({
                    'account_id': f"LOAN_{len(loans) + 1000}",
                    'customer_id': customer['customer_id'],
                    'product_id': product,
                    'open_date': customer['onboarding_date'] + timedelta(days=np.random.randint(0, 365)),
                    'status': 'ACTIVE',
                    'original_amount': loan_amount,
                    'current_balance': loan_amount * np.random.uniform(0.1, 1.0),
                    'interest_rate': np.random.uniform(0.029, 0.15),
                    'term_months': np.random.choice([12, 24, 36, 48, 60, 120]),
                    'monthly_payment': loan_amount * np.random.uniform(0.02, 0.05),
                    'missed_payment_count': np.random.poisson(0.3),
                    'collateral_value': loan_amount * 1.2 if product == 'LOAN_HOME' else 0,
                    'purpose': np.random.choice(['Home Purchase', 'Auto', 'Debt Consolidation', 'Business', 'Personal']),
                    'last_payment_date': self.start_date - timedelta(days=np.random.randint(1, 30)),
                    'next_payment_date': self.start_date + timedelta(days=np.random.randint(1, 30))
                })
        
        return {
            'checking_accounts': pd.DataFrame(checking_accounts),
            'savings_accounts': pd.DataFrame(savings_accounts),
            'credit_cards': pd.DataFrame(credit_cards),
            'loans': pd.DataFrame(loans)
        }

    def generate_transactions(self, accounts: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate detailed transaction history for all account types"""
        print("Generating transaction data...")
        
        checking_transactions = []
        savings_transactions = []
        credit_card_transactions = []
        loan_transactions = []
        
        # Generate checking account transactions
        for _, account in accounts['checking_accounts'].iterrows():
            num_transactions = np.random.poisson(50) # Average 50 transactions per month
            for _ in range(num_transactions * self.num_months):
                transaction_date = self.start_date + timedelta(days=np.random.randint(0, self.num_months * 30))
                merchant = self.merchant_data.sample().iloc[0]
                
                # Apply merchant patterns
                base_amount = np.random.lognormal(np.log(merchant['avg_transaction']), 0.5)
                if transaction_date.weekday() >= 5:  # Weekend
                    base_amount *= merchant['weekend_multiplier']
                
                checking_transactions.append({
                    'transaction_id': f"CHK_TXN_{len(checking_transactions) + 1000}",
                    'account_id': account['account_id'],
                    'transaction_date': transaction_date,
                    'post_date': transaction_date + timedelta(days=np.random.randint(0, 2)),
                    'merchant_id': merchant['merchant_id'],
                    'transaction_type': np.random.choice(['POS', 'ACH', 'CHECK', 'TRANSFER']),
                    'amount': base_amount,
                    'balance_after': np.random.lognormal(8, 1),
                    'status': 'COMPLETED',
                    'description': f"Transaction at {merchant['merchant_name']}"
                })
        
        # Generate savings account transactions
        for _, account in accounts['savings_accounts'].iterrows():
            num_transactions = np.random.poisson(5)  # Fewer transactions for savings
            for _ in range(num_transactions * self.num_months):
                transaction_date = self.start_date + timedelta(days=np.random.randint(0, self.num_months * 30))
                
                savings_transactions.append({
                    'transaction_id': f"SAV_TXN_{len(savings_transactions) + 1000}",
                    'account_id': account['account_id'],
                    'transaction_date': transaction_date,
                    'post_date': transaction_date,
                    'transaction_type': np.random.choice(['DEPOSIT', 'WITHDRAWAL', 'TRANSFER', 'INTEREST']),
                    'amount': np.random.lognormal(6, 1),
                    'balance_after': np.random.lognormal(9, 1),
                    'status': 'COMPLETED',
                    'description': 'Regular savings transaction'
                })
        
        # Generate credit card transactions
        for _, account in accounts['credit_cards'].iterrows():
            num_transactions = np.random.poisson(30)  # Average 30 transactions per month
            for _ in range(num_transactions * self.num_months):
                transaction_date = self.start_date + timedelta(days=np.random.randint(0, self.num_months * 30))
                merchant = self.merchant_data.sample().iloc[0]
                
                credit_card_transactions.append({
                    'transaction_id': f"CC_TXN_{len(credit_card_transactions) + 1000}",
                    'account_id': account['account_id'],
                    'transaction_date': transaction_date,
                    'post_date': transaction_date + timedelta(days=np.random.randint(0, 3)),
                    'merchant_id': merchant['merchant_id'],
                    'amount': np.random.lognormal(4, 1),
                    'status': 'COMPLETED',
                    'is_international': merchant['international'],
                    'mcc_code': merchant['mcc_code'],
                    'rewards_earned': np.random.uniform(1, 5),
                    'description': f"Purchase at {merchant['merchant_name']}"
                })
        
        # Generate loan transactions (payments)
        for _, loan in accounts['loans'].iterrows():
            payment_amount = loan['monthly_payment']
            for month in range(self.num_months):
                payment_date = self.start_date + timedelta(days=30*month + np.random.randint(0, 5))
                
                # Occasionally miss or delay payments
                if np.random.random() < 0.05:  # 5% chance of late payment
                    payment_date += timedelta(days=np.random.randint(1, 30))
                    status = 'LATE'
                else:
                    status = 'COMPLETED'
                
                loan_transactions.append({
                    'transaction_id': f"LOAN_TXN_{len(loan_transactions) + 1000}",
                    'account_id': loan['account_id'],
                    'transaction_date': payment_date,
                    'post_date': payment_date + timedelta(days=1),
                    'amount': payment_amount,
                    'principal_amount': payment_amount * 0.8,
                    'interest_amount': payment_amount * 0.2,
                    'status': status,
                    'remaining_balance': max(0, loan['current_balance'] - (payment_amount * (month + 1))),
                    'description': f"Loan payment for {loan['account_id']}"
                })
        
        return {
            'checking_transactions': pd.DataFrame(checking_transactions),
            'savings_transactions': pd.DataFrame(savings_transactions),
            'credit_card_transactions': pd.DataFrame(credit_card_transactions),
            'loan_transactions': pd.DataFrame(loan_transactions)
        }

    def generate_risk_metrics(self, customers: pd.DataFrame, accounts: Dict[str, pd.DataFrame], 
                            transactions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate risk metrics and scores for customers"""
        print("Generating risk metrics...")
        
        risk_metrics = []
        for _, customer in customers.iterrows():
            # Get customer's accounts and transactions
            customer_cc = accounts['credit_cards'][
                accounts['credit_cards']['customer_id'] == customer['customer_id']]
            customer_loans = accounts['loans'][
                accounts['loans']['customer_id'] == customer['customer_id']]
            
            # Calculate risk metrics
            risk_metrics.append({
                'customer_id': customer['customer_id'],
                'calculation_date': self.start_date + timedelta(days=self.num_months * 30),
                'credit_score': customer['credit_score'],
                'debt_to_income': np.random.uniform(0.2, 0.6),
                'payment_history_score': np.random.uniform(0.7, 1.0),
                'credit_utilization': np.mean(customer_cc['utilization_rate']) if len(customer_cc) > 0 else 0,
                'missed_payments_count': sum(customer_loans['missed_payment_count']) if len(customer_loans) > 0 else 0,
                'risk_score': np.random.uniform(300, 850),
                'risk_category': np.random.choice(['LOW', 'MEDIUM', 'HIGH'], p=[0.6, 0.3, 0.1])
            })
        
        return pd.DataFrame(risk_metrics)

    def save_datasets(self, datasets: Dict[str, pd.DataFrame]):
        """Save all datasets to CSV files in the data directory"""
        print("\nSaving datasets to files...")
        
        for name, df in datasets.items():
            filename = self.data_dir / f"{name}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved {filename} with {len(df)} records")

    def generate_all(self):
        """Generate all banking datasets"""
        # Save static data first
        self.branch_data.to_csv(self.data_dir / "branches.csv", index=False)
        self.product_catalog.to_csv(self.data_dir / "products.csv", index=False)
        self.merchant_data.to_csv(self.data_dir / "merchants.csv", index=False)
        
        # Generate and save main datasets
        customers = self.generate_customers()
        accounts = self.generate_accounts(customers)
        transactions = self.generate_transactions(accounts)
        risk_metrics = self.generate_risk_metrics(customers, accounts, transactions)
        
        # Combine all datasets
        all_datasets = {
            'customers': customers,
            **accounts,
            **transactions,
            'risk_metrics': risk_metrics
        }
        
        # Save all datasets
        self.save_datasets(all_datasets)
        
        return all_datasets

if __name__ == "__main__":
    generator = EnterpriseBankingDataGenerator(
        start_date="2023-01-01",
        num_customers=1000,
        num_months=24
    )
    
    datasets = generator.generate_all()