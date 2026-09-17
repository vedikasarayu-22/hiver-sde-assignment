import os
import sys
import pandas as pd

def download_and_extract_apple_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    apple_csv_path = os.path.join(data_dir, "applesupport_tweets.csv")
    
    if os.path.exists(apple_csv_path):
        print(f"File already exists at: {apple_csv_path}")
        df = pd.read_csv(apple_csv_path)
        print(f"Loaded {len(df)} AppleSupport-related tweets.")
        return apple_csv_path, df

    print("Attempting to download Kaggle dataset 'thoughtvector/customer-support-on-twitter'...")
    try:
        import kagglehub
        dataset_path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
        print(f"Dataset downloaded to: {dataset_path}")
        
        twcs_file = None
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".csv"):
                    twcs_file = os.path.join(root, file)
                    break
        
        if not twcs_file:
            raise FileNotFoundError("Could not find CSV file in downloaded dataset directory.")
            
        print(f"Found CSV file at: {twcs_file}. Filtering for @AppleSupport...")
        
        # Read in chunks to efficiently find AppleSupport tweets
        chunks = []
        for chunk in pd.read_csv(twcs_file, chunksize=100000):
            # Check author_id == 'AppleSupport' or text contains @AppleSupport
            apple_chunk = chunk[
                (chunk['author_id'] == 'AppleSupport') | 
                (chunk['text'].str.contains('@AppleSupport', case=False, na=False))
            ]
            if len(apple_chunk) > 0:
                chunks.append(apple_chunk)
        
        apple_df = pd.concat(chunks, ignore_index=True)
        print(f"Extracted {len(apple_df)} tweets related to @AppleSupport.")
        apple_df.to_csv(apple_csv_path, index=False)
        print(f"Saved to {apple_csv_path}")
        return apple_csv_path, apple_df

    except Exception as e:
        print(f"Error during Kagglehub download: {e}")
        # Secondary fallback: try HuggingFace datasets or direct mirror
        try:
            from datasets import load_dataset
            print("Attempting fallback via HuggingFace datasets...")
            ds = load_dataset("thoughtvector/customer-support-on-twitter", split="train")
            df = ds.to_pandas()
            apple_df = df[
                (df['author_id'] == 'AppleSupport') | 
                (df['text'].str.contains('@AppleSupport', case=False, na=False))
            ]
            print(f"Extracted {len(apple_df)} tweets via HF datasets.")
            apple_df.to_csv(apple_csv_path, index=False)
            return apple_csv_path, apple_df
        except Exception as e2:
            print(f"HF fallback failed: {e2}")
            sys.exit(1)

if __name__ == "__main__":
    download_and_extract_apple_data()
