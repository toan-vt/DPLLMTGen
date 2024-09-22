import pandas as pd
import random

def generate_fake_df(df: pd.DataFrame, num_samples: int = 0, seed: int = 42):
    """
    Generate a fake dataframe with the same columns as the input dataframe
    """
    col_ranges = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        col_ranges[col] = {'dtype': dtype}
        if dtype in ['object', 'category']:
            col_ranges[col]['values'] = list(df[col].unique())
        else:
            col_ranges[col]['min'] = df[col].min()
            col_ranges[col]['max'] = df[col].max()
            col_ranges[col]['values'] = list(df[col].unique())


    random.seed(seed)
    fake_df = []
    if num_samples == 0:
        num_samples = len(df)

    for i in range(num_samples):
        row = {}
        for col in df.columns:
            # row[col] = random.choice(col_ranges[col]['values'])
            if (col_ranges[col]['dtype'] in ['object', 'category']) or len(df[col].unique()) < 10:
                # randomly choose a value from the list of possible values
                row[col] = random.choice(col_ranges[col]['values'])
            else:
                # randomly choose a value from the min-max range
                if 'int' in col_ranges[col]['dtype']:
                    row[col] = random.randint(col_ranges[col]['min'], col_ranges[col]['max'])
                elif 'float' in col_ranges[col]['dtype']:
                    row[col] = random.uniform(col_ranges[col]['min'], col_ranges[col]['max'])
        fake_df.append(row)

    return pd.DataFrame(fake_df)