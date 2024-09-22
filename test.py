from src.DPLMTabGen import DPLMTabGen
from src.preprocessing import *
from src.fake_df_gen import *
from src.metadata import CONFIG
import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="adult", dest="dataset")
parser.add_argument('--seed', type=int, default=42, dest="seed")
parser.add_argument('--target_last', action='store_true')
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf", dest="model_name")
parser.add_argument('--num_epochs', type=int, default=5, dest="num_epochs")
parser.add_argument('--lr', type=float, default=1e-4, dest="lr")
parser.add_argument('--batch-size', type=int, default=0, dest="batch_size")
parser.add_argument('--accum-steps', type=int, default=1, dest="accum_steps")
args = parser.parse_args()

# meta-llama/Llama-2-7b-hf

DATASET = args.dataset
RANDOM_SEED = args.seed
TRAIN_SIZE = 0
EVAL_SIZE = 1
if DATASET == "food":
    EVAL_SIZE = 64
MAX_LENGTH = CONFIG[DATASET]["max_length"]
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

model_name = SHORT_NAME
cache_dir = ".cache"
token = open('/local/scratch/vtran29/keys/llama.token', 'r').read().strip()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token,
                                                cache_dir=f'{cache_dir}/huggingface/{model_name}',
                                                padding_side="left",
                                                trust_remote_code=True)
if 'Llama' in model_name:
    tokenizer.pad_token_id = tokenizer.bos_token_id
elif 'gpt' in model_name:
    tokenizer.pad_token_id = tokenizer.eos_token_id



prompts = dataframe_to_examples(df_train, target_last=False, shuffle=False)
tokenized_prompts = tokinize_examples(prompts, tokenizer, 0)
lengths = [len(x['input_ids']) for x in tokenized_prompts]
print("max length: ", max(lengths))
print("min length: ", min(lengths))
print("avg length: ", sum(lengths)/len(lengths))

print(f"raw prompt: {prompts[0]}")
print(f"formatted ids: {tokenized_prompts[0]['input_ids']}")
print(f"formatted prompt: {tokenizer.decode(tokenized_prompts[0]['input_ids'], skip_special_tokens=False)}")
print("\n")

fake_dataset = get_dataset_from_df(df=df_train, tokenizer=tokenizer, max_length=MAX_LENGTH, target_last=TARGET_LAST, shuffle=False)

# print lengths
print("using separate encoding")
lengths = [len(x['input_ids']) for x in fake_dataset]
print("max length: ", max(lengths))
print("min length: ", min(lengths))
print("avg length: ", sum(lengths)/len(lengths))
# encode input_ids
input_ids = fake_dataset[0]['input_ids']
print("1-stage input_ids: ", input_ids)
print("1-stage decoded: ", tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True))
print("\n")


# lengths = [len(x['input_ids']) for x in tokenized_prompts]
# print("max length: ", max(lengths))
# print("min length: ", min(lengths))
# print("avg length: ", sum(lengths)/len(lengths))

dp_dataset = get_dp_dataset_from_df(df=df_train, tokenizer=tokenizer, max_length=MAX_LENGTH, target_last=TARGET_LAST, shuffle=False)
print(f"dp formatted ids: {dp_dataset[0]['input_ids']}")
print(f"dp formatted prompt: {tokenizer.decode(dp_dataset[0]['input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=True)}")

# for key in fake_dataset[0].keys():
#     print(f"{key}: {len(fake_dataset[0][key])}")

# print("\n\n")
# print(f"29892: |{tokenizer.decode(29892)}|")
# print(f"29893: |{tokenizer.decode(1919)}|")

print(tokenizer.encode("hello world", add_special_tokens=False))
print(tokenizer.encode("hello", add_special_tokens=False))
print(tokenizer.encode(" world", add_special_tokens=False))


print("444, 29871, 22406, 338: ", tokenizer.decode([444, 29871, 22406, 338], skip_special_tokens=False))
print("444, 16767, 338: ", tokenizer.decode([444, 16767, 338], skip_special_tokens=False))