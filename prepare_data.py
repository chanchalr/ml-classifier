import pandas as pd
import requests
import zipfile
import io
import os

def prepare_bank_data():
    # URL for Bank Marketing (Zip format)
    url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
    
    print("Downloading Bank Marketing Dataset...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    # The zip contains 'bank-full.csv' inside another zip or directly. 
    # Let's extract 'bank-full.csv' which has ~45k records.
    z.extractall("temp_bank")
    
    # Specifically, the UCI zip contains 'bank.zip'. Let's unzip that too.
    with zipfile.ZipFile("temp_bank/bank.zip", 'r') as z2:
        z2.extractall("temp_bank")

    # Read the data (Note: UCI uses semicolon as separator for this one)
    df = pd.read_csv("temp_bank/bank-full.csv", sep=';')
    
    # Create directory
    output_dir = 'dataset'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Split into 5 batches of ~9,000 instances to ensure >500 per file
    batch_size = 9000
    for i, start in enumerate(range(0, len(df), batch_size)):
        batch = df.iloc[start : start + batch_size]
        batch_path = os.path.join(output_dir, f'bank_batch_{i+1}.csv')
        batch.to_csv(batch_path, index=False)
        print(f"Created {batch_path} - {len(batch)} rows")
    df.to_csv(os.path.join(output_dir, 'bank_full.csv'), index=False)
    print(f"Created {os.path.join(output_dir, 'bank_full.csv')} - {len(df)} rows")

    print("--- Done! ---")

if __name__ == "__main__":
    prepare_bank_data()