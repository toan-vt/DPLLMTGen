from src.DPLMTabGen import DPLMTabGen
from src.preprocessing import *
from src.fake_df_gen import *
import argparse
import pandas as pd
import torch
from src.metadata import CONFIG

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="adult", dest="dataset")
parser.add_argument('--seed', type=int, default=42, dest="seed")
parser.add_argument('--target_last', action='store_true')
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf", dest="model_name")
parser.add_argument('--ft_model_path', type=str, default=None, dest="ft_model_path")
parser.add_argument('--temperature', type=float, default=0.7, dest="temperature")
parser.add_argument('--csv_output_path', type=str, default="")
parser.add_argument('--batch_size', type=int, default=0, dest="batch_size")

args = parser.parse_args()

print("args: ", args)

DATASET = args.dataset
RANDOM_SEED = args.seed
TRAIN_SIZE = 0
EVAL_SIZE = 512
if DATASET == "food":
    EVAL_SIZE = 64
MAX_LENGTH = CONFIG[DATASET]["max_length"]
if args.batch_size == 0:
    BATCH_SIZE = CONFIG[DATASET]["batch_size"]
else:
    BATCH_SIZE = args.batch_size

MODEL_NAME = args.model_name
SHORT_NAME = MODEL_NAME.split("/")[-1]
TARGET_LAST = args.target_last
FT_MODEL_PATH = args.ft_model_path
TEMPERATURE = args.temperature
OUTPUT_PATH = args.csv_output_path

df_train, df_test = prepare_dataframe(DATASET, random_seed=RANDOM_SEED)
df_train = get_subset_dataframe(df_train, TRAIN_SIZE, RANDOM_SEED)
df_eval = get_subset_dataframe(df_test, EVAL_SIZE, RANDOM_SEED)

token = open('llama.token', 'r').read().strip()

constraint_dict = {}
for col in df_train.columns:
    constraint_dict[col] = {}
    if str(df_train[col].dtype) == "object":
        constraint_dict[col]['type'] = "categorical"
        constraint_dict[col]['unique_values'] = df_train[col].unique().tolist()
    else:
        constraint_dict[col]['type'] = "numerical"
        constraint_dict[col]['min'] = df_train[col].min()
        constraint_dict[col]['max'] = df_train[col].max()

print("constraint_dict: ", constraint_dict)

print("Load pretrained model and start training...")
dptabgen = DPLMTabGen(model_name=MODEL_NAME,
                      cache_dir=".cache",
                      device="cuda:0",
                      token=token,
                      ft_model_path=FT_MODEL_PATH,
    )

# print("Final Format passed: ", dptabgen.format_checking(constraint_dict, num_samples=500, max_length=int(MAX_LENGTH), batch_size=BATCH_SIZE, verbose=True, target_last=TARGET_LAST, wrong_format_verbose=True))

num_samples = TRAIN_SIZE
if num_samples == 0:
    num_samples = len(df_train)

syn_df = dptabgen.sample_dev(constraint_dict, batch_size=BATCH_SIZE, num_samples=100, max_length=MAX_LENGTH, target_last=TARGET_LAST, temperature=TEMPERATURE)
save_file_path = OUTPUT_PATH
if not os.path.exists(os.path.dirname(save_file_path)):
    os.makedirs(os.path.dirname(save_file_path))
syn_df.to_csv(save_file_path, index=False)