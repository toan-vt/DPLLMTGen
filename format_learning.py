from src.DPLMTabGen import DPLMTabGen
from src.preprocessing import *
from src.fake_df_gen import *
from src.metadata import CONFIG
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="adult", dest="dataset")
parser.add_argument('--seed', type=int, default=42, dest="seed")
parser.add_argument('--target_last', action='store_true')
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf", dest="model_name")
parser.add_argument('--num_epochs', type=int, default=5, dest="num_epochs")
parser.add_argument('--lr', type=float, default=1e-4, dest="lr")
parser.add_argument('--batch-size', type=int, default=0, dest="batch_size")
parser.add_argument('--accum-steps', type=int, default=1, dest="accum_steps")
parser.add_argument('--max-length', type=int, default=0, dest="max_length")
args = parser.parse_args()

DATASET = args.dataset
RANDOM_SEED = args.seed
TRAIN_SIZE = 0
EVAL_SIZE = 512
if DATASET == "food":
    EVAL_SIZE = 64
if args.max_length == 0:
    MAX_LENGTH = CONFIG[DATASET]["max_length"]
else:
    MAX_LENGTH = args.max_length
ACCUM_STEPS = args.accum_steps
if args.batch_size == 0:
    BATCH_SIZE = CONFIG[DATASET]["batch_size"]
else:
    BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.lr
MODEL_NAME = args.model_name
TARGET_LAST = args.target_last
SHORT_NAME = MODEL_NAME.split("/")[-1]

print("args: ", args)

print(f"Model: {MODEL_NAME}, Target_last: {TARGET_LAST}, Learning rate: {LEARNING_RATE}, Dataset: {DATASET}, Seed: {RANDOM_SEED}")

df_train, df_test = prepare_dataframe(DATASET, random_seed=RANDOM_SEED)
df_train = get_subset_dataframe(df_train, TRAIN_SIZE, RANDOM_SEED)
df_eval = get_subset_dataframe(df_test, EVAL_SIZE, RANDOM_SEED)
df_fake = generate_fake_df(df=df_train, num_samples=TRAIN_SIZE, seed=RANDOM_SEED)

token = open('llama.token', 'r').read().strip()

dptabgen = DPLMTabGen(model_name=MODEL_NAME,
                      cache_dir=".cache",
                      device="cuda:0",
                      token=token,
                    #   ft_model_path="models/Llama-2-7b-chat-hf_TargetLast-True_0.0001/sick/0_42/p1_final",
    )

fake_dataset = get_dataset_from_df(df=df_fake, tokenizer=dptabgen.tokenizer, max_length=MAX_LENGTH, target_last=TARGET_LAST)
eval_fake_dataset = get_dataset_from_df(df=df_eval, tokenizer=dptabgen.tokenizer, max_length=MAX_LENGTH, target_last=TARGET_LAST)

print("Starting Format Learning ========================================================================================")
dptabgen.format_learning_tune(fake_dataset,
                                fake_eval_dataset = eval_fake_dataset,
                                num_epochs = NUM_EPOCHS,
                                batch_size = BATCH_SIZE,
                                accum_steps = ACCUM_STEPS,
                                lr=LEARNING_RATE,
                                save_log=True,
                                log_file=f"./log/{DATASET}/{SHORT_NAME}_TargetLast-{TARGET_LAST}_{LEARNING_RATE}_{TRAIN_SIZE}_{RANDOM_SEED}.txt",
                                save_model_dir=f"./models/{DATASET}/{SHORT_NAME}_TargetLast-{TARGET_LAST}_{LEARNING_RATE}/{TRAIN_SIZE}_{RANDOM_SEED}",
                                save_every_epoch=True,
                                verbose=True
                            )
print("Finish Format Learning ========================================================================================")

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

print("Format passed: ", dptabgen.format_checking(constraint_dict, num_samples=500, max_length=MAX_LENGTH, batch_size=BATCH_SIZE, verbose=True, target_last=TARGET_LAST, wrong_format_verbose=True))
