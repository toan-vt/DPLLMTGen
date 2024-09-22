import pandas as pd
from datasets import Dataset
import random
import numpy as np
from sklearn.model_selection import train_test_split
import openml
import os
TASK_DESCRIPTION = "## "
SPECIAL_TOKEN = "&"

def prepare_dataframe(dataset, random_seed: int = 42):
    if dataset == 'adult':
        df = pd.read_csv("data/adult/clean_adult.csv")
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_seed)
        # strip all categorical columns
        for col in df_train.columns:
            if df_train[col].dtype == 'object':
                df_train[col] = df_train[col].str.strip()
                df_test[col] = df_test[col].str.strip()

        if not os.path.exists(f"./data/adult/adult_train_{random_seed}.csv"):
            df_train.to_csv(f"./data/adult/adult_train_{random_seed}.csv", index=False)
            df_test.to_csv(f"./data/adult/adult_test_{random_seed}.csv", index=False)
            
        return df_train, df_test

    elif dataset == 'shipping':
        df = pd.read_csv("./data/shipping/shipping.csv")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_seed)
        if not os.path.exists(f"./data/shipping/shipping_train_{random_seed}.csv"):
            df_train.to_csv(f"./data/shipping/shipping_train_{random_seed}.csv", index=False)
            df_test.to_csv(f"./data/shipping/shipping_test_{random_seed}.csv", index=False)
        return df_train, df_test
    elif dataset == 'healthcare':
        df = pd.read_csv("./data/healthcare/healthcare.csv")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_seed)
        if not os.path.exists(f"./data/healthcare/healthcare_train_{random_seed}.csv"):
            df_train.to_csv(f"./data/healthcare/healthcare_train_{random_seed}.csv", index=False)
            df_test.to_csv(f"./data/healthcare/healthcare_test_{random_seed}.csv", index=False)

        return df_train, df_test

    else:
        AssertionError("dataset must be either 'adult'")

def get_subset_dataframe(df, num_samples: int, random_seed: int = 42):
    if num_samples > len(df):
        print("ERROR: num_samples must be less than or equal to the number of samples in the dataframe")
        return df
    elif num_samples == 0:
        return df
    return df.sample(n=num_samples, random_state=random_seed)

def dataframe_to_examples(df, random_seed=42, target_last=False, shuffle=True):
    random.seed(random_seed)
    examples = []
    for index, row in df.iterrows():
        if not target_last:
            arr = list(range(len(df.columns)))
            if shuffle:
                random.shuffle(arr)
        else:
            arr = list(range(len(df.columns) - 1))
            if shuffle:
                random.shuffle(arr)
            arr.append(len(df.columns) - 1)
        prompt = str(TASK_DESCRIPTION)
        for col_idx in arr:
            prompt += f"{df.columns[col_idx]} is {row[df.columns[col_idx]]}, "
        prompt = prompt[:-2]
        prompt = ' '.join(prompt.split())
        examples.append({'text': prompt})

    return examples

def tokinize_examples(examples, tokenizer, max_length: int = 128):
    if max_length == 0:
        tokenized_prompts = [tokenizer(example['text'] + tokenizer.eos_token, add_special_tokens=False) for example in examples]
    else:
        # tokenize without adding a end of sentence token
        tokenized_prompts = [tokenizer(example['text'] + tokenizer.eos_token, add_special_tokens=False, 
                                    padding="max_length", max_length=max_length) for example in examples]
    for i in range(len(tokenized_prompts)):
        num_paddding_token = (np.array(tokenized_prompts[i]['attention_mask']) == 0).sum()
        tokenized_prompts[i]['labels'] = [-100]*num_paddding_token + tokenized_prompts[i]['input_ids'][num_paddding_token:]

    return tokenized_prompts

# def get_dataset_from_df(df, tokenizer, max_length: int = 128, target_last=False, shuffle=True):
#     prompts = dataframe_to_examples(df, target_last=target_last, shuffle=shuffle)
#     tokenized_prompts = tokinize_examples(prompts, tokenizer, max_length)
#     return Dataset.from_list(tokenized_prompts)

def get_dataset_from_df(df, tokenizer, max_length: int = 128, target_last=False, shuffle=True):
    examples = []
    for index, row in df.iterrows():
        if not target_last:
            arr = list(range(len(df.columns)))
            if shuffle:
                random.shuffle(arr)
        else:
            arr = list(range(len(df.columns) - 1))
            if shuffle:
                random.shuffle(arr)
            arr.append(len(df.columns) - 1)
        input_ids = tokenizer.encode(TASK_DESCRIPTION, add_special_tokens=False)
        attention_mask = [1]*len(input_ids)
        for i, col_idx in enumerate(arr):
            if i == 0:
                s_ = f'{SPECIAL_TOKEN}{df.columns[col_idx]} is'
            else:
                s_ = f'{SPECIAL_TOKEN}, {df.columns[col_idx]} is'

            col_str_ids = tokenizer.encode(s_, add_special_tokens=False)[1:]
            input_ids.extend(col_str_ids)
            attention_mask.extend([1] * len(col_str_ids))

            val_ids = tokenizer.encode(f"{SPECIAL_TOKEN} {row[df.columns[col_idx]]}", add_special_tokens=False)[1:]
            if i == (len(arr) - 1):
                val_ids = val_ids + [tokenizer.eos_token_id]
            input_ids.extend(val_ids)
            attention_mask.extend([1] * len(val_ids))

        labels = input_ids.copy()
        labels = [-100]* (max_length - len(labels)) + labels
        input_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids))  + input_ids
        attention_mask = [0] * (max_length - len(attention_mask)) + attention_mask
        
        examples.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels})
    
    return Dataset.from_list(examples)

def get_dp_dataset_from_df(df, tokenizer, max_length: int = 128, target_last=False, shuffle=True):      
    examples = []
    for index, row in df.iterrows():
        if not target_last:
            arr = list(range(len(df.columns)))
            if shuffle:
                random.shuffle(arr)
        else:
            arr = list(range(len(df.columns) - 1))
            if shuffle:
                random.shuffle(arr)
            arr.append(len(df.columns) - 1)
        input_ids = tokenizer.encode(TASK_DESCRIPTION, add_special_tokens=False)
        attention_mask = [1]*len(input_ids)
        is_labels = [False]*len(input_ids)
        is_numbers = [False]*len(input_ids)
        is_targets = [False]*len(input_ids)
        # num_columns = [-1]*len(input_ids)
        num_columns = []
        for i, col_idx in enumerate(arr):
            if i == 0:
                s_ = f'{SPECIAL_TOKEN}{df.columns[col_idx]} is'
            else:
                s_ = f'{SPECIAL_TOKEN}, {df.columns[col_idx]} is'

            col_str_ids = tokenizer.encode(s_, add_special_tokens=False)[1:]
            
            # print(f'\tstr: "{s_}"')
            # print(f'\tcol_str_ids: "{col_str_ids}"')
            # print(f'\tdecoded: "{tokenizer.decode(col_str_ids, skip_special_tokens=False)}"\n')
            

            input_ids.extend(col_str_ids)
            attention_mask.extend([1] * len(col_str_ids))
            is_labels.extend([False] * len(col_str_ids))
            is_numbers.extend([False] * len(col_str_ids))
            is_targets.extend([False] * len(col_str_ids))
            # num_columns.extend([-1] * len(col_str_ids))

            val_ids = tokenizer.encode(f"{SPECIAL_TOKEN} {row[df.columns[col_idx]]}", add_special_tokens=False)[1:]
            # print(f'\tval: "{row[df.columns[col_idx]]}"')
            # print(f'\tval_ids: "{val_ids}"')
            # print(f'\tdecoded: "{tokenizer.decode(val_ids, skip_special_tokens=False)}"\n')
            if i == (len(arr) - 1):
                val_ids = val_ids + [tokenizer.eos_token_id]
            input_ids.extend(val_ids)
            attention_mask.extend([1] * len(val_ids))
            is_labels.extend([True] * len(val_ids))
            if (df[df.columns[col_idx]].dtype in ['int64', 'float64']):
                is_numbers.extend([True] * len(val_ids))
                # num_columns.extend([col_idx] * len(val_ids))
                num_columns.append(col_idx)
            else:
                is_numbers.extend([False] * len(val_ids))
                # num_columns.extend([-1]*len(val_ids))
            
            if i == (len(arr) - 1):
                is_targets.extend([True] * len(val_ids))

        labels = input_ids.copy()
        labels = [-100]* (max_length - len(labels)) + labels
        input_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids))  + input_ids
        attention_mask = [0] * (max_length - len(attention_mask)) + attention_mask
        is_labels = [False] * (max_length - len(is_labels)) + is_labels
        is_numbers = [False] * (max_length - len(is_numbers)) + is_numbers
        is_targets = [False] * (max_length - len(is_targets)) + is_targets
        # num_columns = [-1] * (max_length - len(num_columns)) + num_columns
        
        examples.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'is_labels': is_labels, 'is_numbers': is_numbers, 'is_targets': is_targets, 'num_columns': num_columns})
    
    return Dataset.from_list(examples)

