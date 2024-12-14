import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class BankingDataGenerator:
    def __init__(self, start_date="2023-01-01", num_customers=1000, num_months=24):
        """
        Initialize the banking data generator with parameters
        
        Parameters:
        - start_date: Starting date for transaction history
        - num_customers: Number of customers to generate
        - num_months: Number of months of history to generate
        """
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.num_customers = num_customers
        self.num_months = num_months
        self.date_range = pd.date_range(start=self.start_date, 
                                      periods=num_months * 30, 
                                      freq='D')
        
    def generate_customer_profiles(self):
        """Generate base customer demographic and financial profiles"""
        np.random.seed(42)
        
        profiles = pd.DataFrame({
            'customer_id': range(1, self.num_customers + 1),
            'age': np.random.normal(40, 12, self.num_customers).clip(21, 75).astype(int),
            'income': np.random.lognormal(11, 0.5, self.num_customers).clip(30000, 250000),
            'employment_length': np.random.normal(8, 4, self.num_customers).clip(0, 40),
            'credit_score': np.random.normal(700, 80, self.num_customers).clip(450, 850),
            'num_credit_cards': np.random.poisson(2, self.num_customers).clip(1, 5),
            'mortgage_amount': np.random.lognormal(12, 1, self.num_customers).clip(0, 1000000),
            'initial_savings': np.random.lognormal(10, 1, self.num_customers).clip(1000, 500000)
        })
        
        # Add categorical features
        profiles['education'] = np.random.choice(
            ['High School', 'Bachelor', 'Master', 'PhD'],
            size=self.num_customers,
            p=[0.3, 0.4, 0.25, 0.05]
        )
        
        profiles['occupation_sector'] = np.random.choice(
            ['Technology', 'Healthcare', 'Finance', 'Education', 'Other'],
            size=self.num_customers,
            p=[0.25, 0.2, 0.15, 0.15, 0.25]
        )
        
        return profiles

    def generate_monthly_patterns(self, customer_profiles):
        """Generate monthly financial patterns for each customer"""
        all_monthly_data = []
        
        for _, customer in customer_profiles.iterrows():
            # Base monthly income (salary)
            monthly_income = customer['income'] / 12
            
            # Generate monthly patterns
            for month in range(self.num_months):
                # Add some random variation to monthly income
                current_income = monthly_income * np.random.normal(1, 0.1)
                
                # Calculate monthly expenses based on income and random variation
                housing_expense = customer['mortgage_amount'] * 0.005 if customer['mortgage_amount'] > 0 else current_income * 0.3
                living_expenses = current_income * np.random.uniform(0.2, 0.4)
                discretionary_spending = current_income * np.random.uniform(0.1, 0.3)
                
                # Credit card behavior
                credit_utilization = np.random.beta(2, 5) * 100  # Generally lower utilization
                
                # Savings behavior with seasonal patterns
                base_savings_rate = np.random.beta(2, 3)  # Base savings rate
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * month / 12)  # Seasonal variation
                savings_amount = current_income * base_savings_rate * seasonal_factor
                
                # Add some randomness to payment behavior
                late_payment = np.random.choice([0, 1], p=[0.95, 0.05])
                
                monthly_data = {
                    'customer_id': customer['customer_id'],
                    'month': month,
                    'date': self.start_date + timedelta(days=30*month),
                    'monthly_income': current_income,
                    'housing_expense': housing_expense,
                    'living_expenses': living_expenses,
                    'discretionary_spending': discretionary_spending,
                    'credit_utilization': credit_utilization,
                    'savings_amount': savings_amount,
                    'late_payment': late_payment,
                }
                
                all_monthly_data.append(monthly_data)
        
        return pd.DataFrame(all_monthly_data)

    def generate_transaction_data(self, customer_profiles, monthly_patterns):
        """Generate detailed daily transaction data"""
        all_transactions = []
        
        for _, customer in customer_profiles.iterrows():
            customer_monthly = monthly_patterns[
                monthly_patterns['customer_id'] == customer['customer_id']
            ]
            
            for _, month_data in customer_monthly.iterrows():
                # Generate number of transactions for the month
                num_transactions = np.random.poisson(30)  # Average 30 transactions per month
                
                for _ in range(num_transactions):
                    # Generate transaction date within the month
                    transaction_date = month_data['date'] + timedelta(
                        days=np.random.randint(0, 30)
                    )
                    
                    # Generate transaction type and amount
                    transaction_type = np.random.choice([
                        'grocery', 'shopping', 'entertainment', 'utility',
                        'healthcare', 'travel', 'restaurant', 'transfer'
                    ], p=[0.3, 0.15, 0.1, 0.1, 0.05, 0.1, 0.15, 0.05])
                    
                    # Base amount depends on transaction type
                    base_amounts = {
                        'grocery': 100,
                        'shopping': 200,
                        'entertainment': 150,
                        'utility': 300,
                        'healthcare': 500,
                        'travel': 1000,
                        'restaurant': 80,
                        'transfer': 500
                    }
                    
                    # Add randomness to amount
                    amount = np.random.lognormal(
                        np.log(base_amounts[transaction_type]),
                        0.5
                    )
                    
                    transaction = {
                        'customer_id': customer['customer_id'],
                        'date': transaction_date,
                        'transaction_type': transaction_type,
                        'amount': amount,
                        'balance_after_transaction': None  # Will be calculated later
                    }
                    
                    all_transactions.append(transaction)
        
        transactions_df = pd.DataFrame(all_transactions)
        transactions_df = transactions_df.sort_values(['customer_id', 'date'])
        
        return transactions_df

    def calculate_default_risk(self, customer_profiles, monthly_patterns, transactions):
        """Calculate loan default risk based on complex patterns"""
        default_risks = []
        
        for customer_id in customer_profiles['customer_id']:
            customer = customer_profiles[customer_profiles['customer_id'] == customer_id].iloc[0]
            customer_monthly = monthly_patterns[monthly_patterns['customer_id'] == customer_id]
            customer_transactions = transactions[transactions['customer_id'] == customer_id]
            
            # Calculate risk factors
            
            # 1. Income stability
            income_volatility = customer_monthly['monthly_income'].std() / customer_monthly['monthly_income'].mean()
            
            # 2. Savings trend (last 6 months vs first 6 months)
            savings_trend = (
                customer_monthly['savings_amount'].iloc[-6:].mean() /
                customer_monthly['savings_amount'].iloc[:6].mean()
            )
            
            # 3. Credit utilization trend
            credit_util_trend = (
                customer_monthly['credit_utilization'].iloc[-3:].mean() -
                customer_monthly['credit_utilization'].iloc[:3].mean()
            )
            
            # 4. Late payment history
            late_payment_ratio = customer_monthly['late_payment'].mean()
            
            # 5. Expense ratio trend
            expense_ratio = (
                (customer_monthly['housing_expense'] + 
                 customer_monthly['living_expenses'] + 
                 customer_monthly['discretionary_spending']) / 
                customer_monthly['monthly_income']
            ).mean()
            
            # 6. Transaction pattern changes
            recent_transaction_volume = len(customer_transactions[
                customer_transactions['date'] >= self.date_range[-90]
            ])
            
            # Calculate default probability
            default_prob = (
                0.3 * income_volatility +
                0.2 * (1 - min(savings_trend, 1)) +
                0.15 * (credit_util_trend / 100) +
                0.2 * late_payment_ratio +
                0.1 * (expense_ratio - 0.7) +
                0.05 * (recent_transaction_volume / 90)
            ).clip(0, 1)
            
            # Add some noise to prevent perfect correlation
            default_prob = np.random.beta(
                default_prob * 10,
                (1 - default_prob) * 10
            )
            
            default_risks.append({
                'customer_id': customer_id,
                'default_probability': default_prob,
                'is_default': int(default_prob > 0.5)
            })
        
        return pd.DataFrame(default_risks)

    def generate_complete_dataset(self):
        """Generate complete banking dataset"""
        print("Generating customer profiles...")
        customer_profiles = self.generate_customer_profiles()
        
        print("Generating monthly patterns...")
        monthly_patterns = self.generate_monthly_patterns(customer_profiles)
        
        print("Generating transaction data...")
        transactions = self.generate_transaction_data(customer_profiles, monthly_patterns)
        
        print("Calculating default risks...")
        default_risks = self.calculate_default_risk(customer_profiles, monthly_patterns, transactions)
        
        return {
            'customer_profiles': customer_profiles,
            'monthly_patterns': monthly_patterns,
            'transactions': transactions,
            'default_risks': default_risks
        }

# Usage example
if __name__ == "__main__":
    generator = BankingDataGenerator(
        start_date="2023-01-01",
        num_customers=1000,
        num_months=24
    )
    
    dataset = generator.generate_complete_dataset()
    
    # Save to CSV files
    for name, df in dataset.items():
        df.to_csv(f"{name}.csv", index=False)
        print(f"Saved {name}.csv with {len(df)} records")