from src.evaluate import *
from src.preprocessing import *
import argparse
import warnings
import time
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, default="test.csv", dest="test_file")
parser.add_argument('--syn_file', type=str, default="syn.csv", dest="syn_file")
parser.add_argument('--seed', type=int, default=42, dest="seed")
parser.add_argument('--kway', type=int, default=4, dest="kway")
parser.add_argument('--fairness', type=bool, default=False, dest="fairness")
args = parser.parse_args()

print("ARGS: ", args)

TEST_FILE = args.test_file
SYNTHETIC_FILE = args.syn_file
RANDOM_SEED = args.seed
KWAY = args.kway + 1
OTHER_METRICS = True
FAIRNESS = args.fairness
df_test = pd.read_csv(TEST_FILE)
df_synthetic = pd.read_csv(SYNTHETIC_FILE)
df_synthetic = df_synthetic.fillna('nan')

# strip all df_synthetic cols
for col in df_synthetic.columns:
    if df_synthetic[col].dtype == 'object':
        df_synthetic[col] = df_synthetic[col].str.strip()

df_test_1, df_synthetic_1 = to_numbers(df_test, df_synthetic, n_bins=20, categorical_only=False)
tvd_1 = get_tvd(df_test_1, df_synthetic_1)
print("TVD 1: ", tvd_1['tvd'].mean())

if KWAY > len(df_test.columns) + 1:
    KWAY = len(df_test.columns) + 1

for k in range(2, KWAY):
    now = time.time()
    df_test_k, df_synthetic_k = to_k_margin(df_test_1, df_synthetic_1, k)
    tvd = get_tvd(df_test_k, df_synthetic_k)
    print(f"TVD {k}: ", tvd['tvd'].mean(), time.time() - now)

# df_test_2, df_synthetic_2 = to_2_margin(df_test_1, df_synthetic_1)
# tvd_2 = get_tvd(df_test_2, df_synthetic_2)
# print("TVD 2: ", tvd_2['tvd'].mean())

# now = time.time()
# df_test_3, df_synthetic_3 = to_3_margin(df_test_1, df_synthetic_1)
# now_1 = time.time()
# tvd_3 = get_tvd(df_test_3, df_synthetic_3)
# now_2 = time.time()
# print(now_1 - now, now_2 - now_1, now_2 - now)
# print("TVD 3: ", tvd_3['tvd'].mean())

# now = time.time()
# df_test_3, df_synthetic_3 = to_k_margin(df_test_1, df_synthetic_1, k=3)
# now_1 = time.time()
# tvd_3 = get_tvd(df_test_3, df_synthetic_3)
# now_2 = time.time()
# print(now_1 - now, now_2 - now_1, now_2 - now)
# print("TVD 3: ", tvd_3['tvd'].mean())

print("XGB Performance: ", get_xgboost_performance(df_test, df_synthetic, others=OTHER_METRICS, fairness_metrics=FAIRNESS))
print("LR Performance: ", get_lr_performance(df_test, df_synthetic, others=OTHER_METRICS))
