import pandas as pd


def load_clean_telco_data(data_path):
    """Load the Telco CSV and apply standard cleaning.
    
    - Coerces TotalCharges to float, imputing 0 for the 11 tenure=0 customers
    - Returns a clean DataFrame ready for analysis
    """
    df = pd.read_csv(data_path)
    df.loc[df['TotalCharges'] == ' ', 'TotalCharges'] = '0'
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    return df.copy()