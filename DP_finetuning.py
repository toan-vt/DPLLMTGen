from src.DPLMTabGen import DPLMTabGen
from src.preprocessing import *
from src.fake_df_gen import *
import argparse
import pandas as pd
import torch
from src.metadata import CONFIG

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="credit", dest="dataset")
parser.add_argument('--epsilon', type=float, default=4.0, dest="epsilon")
parser.add_argument('--lr', type=float, default=1e-3, dest="learning_rate")
parser.add_argument('--training_size', type=int, default=0, dest="training_size")
parser.add_argument('--num_dp_epochs', type=int, default=3, dest="num_dp_epochs")
parser.add_argument('--alpha', type=float, default=0.9, dest="alpha")
parser.add_argument('--beta', type=float, default=1.0, dest="beta")
parser.add_argument('--target_last', action='store_true')
parser.add_argument('--seed', type=int, default=42, dest="seed")
parser.add_argument('--numerical_loss', action='store_true')
parser.add_argument('--gamma', type=float, default=0.0, dest="gamma")
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf", dest="model_name")
parser.add_argument('--batch_size', type=int, default=0, dest="batch_size")
parser.add_argument('--accum_steps', type=int, default=1, dest="accum_steps")
parser.add_argument('--formatted_model_path', type=str, default=None, dest="formatted_model_path")
parser.add_argument('--fastdp', action='store_true')
parser.add_argument('--temperature', type=float, default=1.0, dest="temperature")
parser.add_argument('--max_length', type=int, default=0, dest="max_length")
args = parser.parse_args()

DATASET = args.dataset
RANDOM_SEED = args.seed
TRAIN_SIZE = args.training_size
EVAL_SIZE = 512
if DATASET == "food":
    EVAL_SIZE = 64
if args.max_length == 0:
    MAX_LENGTH = CONFIG[DATASET]["max_length"]
else:
    MAX_LENGTH = args.max_length
NUM_DP_EPOCHS = args.num_dp_epochs
LEARNING_RATE = args.learning_rate
ALPHA = args.alpha
BETA = args.beta
GAMMA = args.gamma
EPSILON = args.epsilon
MODEL_NAME = args.model_name
SHORT_NAME = MODEL_NAME.split("/")[-1]
TEMPERATURE = args.temperature
# if '7' in MODEL_NAME:
#     BATCH_SIZE = CONFIG[DATASET]["batch_size"]
# elif '13' in MODEL_NAME:
#     BATCH_SIZE = 16

if args.batch_size == 0:
    BATCH_SIZE = CONFIG[DATASET]["batch_size"]
else:
    BATCH_SIZE = args.batch_size
ACCUM_STEPS = args.accum_steps
TARGET_LAST = args.target_last
NUMERICAL_LOSS = args.numerical_loss
TEMPERATURE = args.temperature
FT_MODEL_PATH = args.formatted_model_path
FASTDP = args.fastdp

print("args: ", args)

# #TODO hardcode, need to fix
# if "Mistral" in MODEL_NAME: # Mistral model
#     FT_MODEL_PATH = CONFIG[DATASET]["ft_model_path_Mistral_noTL"]
# elif "-chat-" not in MODEL_NAME: # llama-2-7b-hf
#     FT_MODEL_PATH = CONFIG[DATASET]["ft_model_path_nonChat"]
# elif TARGET_LAST: # llama-2-7b-chat-hf with target last
#     FT_MODEL_PATH = CONFIG[DATASET]["ft_model_path"]
# else: # llama-2-7b-chat-hf without target last
#     FT_MODEL_PATH = CONFIG[DATASET]["ft_model_path_noTL"] # llama-2-7b-chat-hf without target last
#     if '13b' in MODEL_NAME: # llama-2-13b-chat-hf without target last
#         FT_MODEL_PATH = CONFIG[DATASET]["ft_model_path_13b_noTL"]
    
df_train, df_test = prepare_dataframe(DATASET, random_seed=RANDOM_SEED)
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

df_train = get_subset_dataframe(df_train, TRAIN_SIZE, RANDOM_SEED)
df_eval = get_subset_dataframe(df_test, EVAL_SIZE, RANDOM_SEED)

token = open('llama.token', 'r').read().strip()

print(f"Model Name: {MODEL_NAME}")
print(f"FT Model Name: {FT_MODEL_PATH}")
print(f"TARGET_LAST: {TARGET_LAST}, NUMERICAL_LOSS: {NUMERICAL_LOSS}\n\n")

print("Load pretrained model and start training...")
dptabgen = DPLMTabGen(model_name=MODEL_NAME,
                      cache_dir=".cache",
                      device="cuda:0",
                      token=token,
                      ft_model_path=FT_MODEL_PATH,
    )

# print("Checking format learning")
# print("Format passed: ", dptabgen.format_checking(constraint_dict, num_samples=500, max_length=MAX_LENGTH, batch_size=BATCH_SIZE, verbose=True, target_last=TARGET_LAST))

train_dataset = get_dp_dataset_from_df(df=df_train, tokenizer=dptabgen.tokenizer, max_length=MAX_LENGTH, target_last=TARGET_LAST)
eval_dataset = get_dp_dataset_from_df(df=df_eval, tokenizer=dptabgen.tokenizer, max_length=MAX_LENGTH, target_last=TARGET_LAST)

max_abs_values = []
for idx, col in enumerate(df_train.columns):
    if df_train[col].dtype in ['int64', 'float64']:
        max_abs_values.append(max(abs(df_train[col].min()), abs(df_train[col].max())))
    else:
        max_abs_values.append(0)


# print("Starting DP Learning ========================================================================================")
dptabgen.dp_tune(
    train_dataset,
    eval_dataset,
    num_epochs=NUM_DP_EPOCHS,
    batch_size=BATCH_SIZE,
    accum_steps=ACCUM_STEPS,
    epsilon=EPSILON,
    alpha=ALPHA,
    numerical_loss=NUMERICAL_LOSS,
    beta=BETA,
    gamma=GAMMA,
    max_abs_values=max_abs_values,
    lr=LEARNING_RATE,
    save_log=True,
    log_file=f"./log/{DATASET}/{SHORT_NAME}_TL{TARGET_LAST}{GAMMA}_NL{NUMERICAL_LOSS}_{TRAIN_SIZE}_{RANDOM_SEED}_{EPSILON}_{NUM_DP_EPOCHS}_{ALPHA}_{LEARNING_RATE}.txt",
    save_model_dir=f"./dp-models/{DATASET}/{SHORT_NAME}_TL{TARGET_LAST}{GAMMA}_NL{NUMERICAL_LOSS}/{TRAIN_SIZE}_{RANDOM_SEED}_{EPSILON}_{NUM_DP_EPOCHS}_{ALPHA}_{LEARNING_RATE}",
    save_every_epoch=True,
    verbose=True,
    fastdp=FASTDP
)
# print("Finishing DP Learning ======================================================================================")

num_samples = TRAIN_SIZE
if num_samples == 0:
    num_samples = len(df_train)

syn_df = dptabgen.sample(constraint_dict, batch_size=BATCH_SIZE, num_samples=num_samples, max_length=MAX_LENGTH*2, target_last=TARGET_LAST, temperature=TEMPERATURE)
save_file_path = f"./data/{DATASET}/synthetic/{SHORT_NAME}_TL{TARGET_LAST}{GAMMA}_NL{NUMERICAL_LOSS}_{TRAIN_SIZE}_{RANDOM_SEED}_{EPSILON}_{NUM_DP_EPOCHS}_{ALPHA}_{LEARNING_RATE}_TEM{TEMPERATURE}.csv"
if not os.path.exists(os.path.dirname(save_file_path)):
    os.makedirs(os.path.dirname(save_file_path))
syn_df.to_csv(save_file_path, index=False)