import sys
import pandas as pd
import numpy as np
import random
import os
import json
import pdb
from datasets import load_dataset

num_clients = int(sys.argv[1])
diff_quantity = int(sys.argv[2])
dataset_type = sys.argv[3]

MAX_PER_TASK = 210 
alpha = 1


np.random.seed(42)
random.seed(42)

if dataset_type.lower() == "dolly":
    print(">>> Loading Dolly dataset ...")
    df = pd.read_json("../new-databricks-dolly-15k.json", orient='records')

elif dataset_type.lower() == "mmlu":
    print(">>> Loading MMLU dataset ...")
    base_path = "../mmlu_all_tasks"
    
    rows = []
    split_files = ["dev.json", "validation.json", "test.json"] 
    for subject in os.listdir(base_path):
        subject_dir = os.path.join(base_path, subject)
        if not os.path.isdir(subject_dir):
            continue
        for fname in split_files:
            fpath = os.path.join(subject_dir, fname)
            if not os.path.exists(fpath):
                continue

            with open(fpath, "r") as f:
                for line in f:
                    data = json.loads(line)

                    instruction = data["question"]
                    choices = data["choices"]
                    answer_idx = data["answer"]
                    answer_text = None

                    formatted_instruction = instruction + "\nAnswer choices:\n"
                    for i, c in enumerate(choices):
                        formatted_instruction += f"{chr(ord('A')+i)}: {c}\n"
                        if answer_idx == i:
                            answer_text = f"{chr(ord('A')+i)}: {c}"

                    rows.append({
                        "instruction": formatted_instruction,
                        "context": "",
                        "response": answer_text,
                        "category": data["subject"],
                    })
    df = pd.DataFrame(rows)

    subject_counts = df["category"].value_counts()
    print(f">>> Total MMLU subjects found: {len(subject_counts)}")

    df = (
        df.groupby("category", group_keys=False)
          .apply(lambda x: x.sample(n=min(MAX_PER_TASK, len(x)), random_state=42))
          .reset_index(drop=True)
    )

    print(f">>> Total MMLU subjects found: {len(subject_counts)}")
    print(f">>> Subjects after {MAX_PER_TASK}-cap: {df['category'].nunique()}")
    print(">>> Sample count per subject (after cap):")
    print(df['category'].value_counts().sort_index())


elif dataset_type.lower() == "glue":
    print(">>> Loading GLUE dataset (all tasks) ...")
    glue_tasks = ["sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "cola"]

    rows = []

    glue_questions = {
        "sst2": "What is the sentiment? Positive or Negative?",
        "mrpc": "Do both sentences say the same thing? Yes or No?",
        "qqp": "Do both questions ask the same thing? Yes or No?",
        "qnli": "Does the sentence answer the question? Yes or No?",
        "rte": "True or False?",
        "mnli": "Is the second sentence an Entailment, Contradiction, or Neutral?",
        "cola": "Is this sentence linguistically acceptable? Yes or No?",
    }

    glue_label_map = {
        "sst2": {0: "Negative", 1: "Positive"},
        "mrpc": {0: "No", 1: "Yes"},
        "qqp": {0: "No", 1: "Yes"},
        "qnli": {0: "Yes", 1: "No"},
        "rte": {0: "False", 1: "True"},
        "mnli": {0: "Entailment", 1: "Neutral", 2: "Contradiction"},
        "cola": {0: "No", 1: "Yes"},
    }

    for task in glue_tasks:
        print(f"--- Loading GLUE task: {task} ---")
        try:
            ds = load_dataset("glue", task)
        except:
            print(f"[WARNING] Failed to load GLUE task: {task}")
            continue

        split = "train" if "train" in ds else ds.keys()[0]
        data = ds[split]

        for item in data:
            question = glue_questions[task]
            label_str = glue_label_map[task][item["label"]]

            if task in ["sst2", "cola"]:
                s1 = item["sentence"]
                s2 = None
            elif task == "mrpc":
                s1 = item["sentence1"]
                s2 = item["sentence2"]
            elif task == "qqp":
                s1 = item["question1"]
                s2 = item["question2"]
            elif task == "qnli":
                s1 = item["question"]
                s2 = item["sentence"]
            elif task == "rte":
                s1 = item["sentence1"]
                s2 = item["sentence2"]
            elif task == "mnli":
                s1 = item["premise"]
                s2 = item["hypothesis"]
            else:
                continue

            if s2 is not None:
                query = f"sentence1: {s1}\n"
                query += f"sentence2: {s2}\n"
            else:
                query = f"sentence: {s1}\n"

            query += f"Question: answer with one word. {question}\n"

            rows.append({
                "instruction": query,
                "context": "",
                "response": label_str,
                "category": task
            })

    df = pd.DataFrame(rows)
    df = df.groupby("category").apply(lambda x: x.sample(n=min(610, len(x)), random_state=42)).reset_index(drop=True)

else:
    raise ValueError("dataset_type must be either 'dolly' or 'truthful'")



sorted_df = df.sort_values(by=['category'])


grouped = sorted_df.groupby('category')

test_df = grouped.apply(lambda x: x.sample(n=100))
test_df = test_df.reset_index(level=0, drop=True)

test10_df = test_df.groupby("category").apply(lambda x: x.sample(n=10))
test10_df = test10_df.reset_index(level=0, drop=True)

valid_df = grouped.apply(
    lambda x: x.drop(

        x.index.intersection(test_df.index)
    ).sample(n=10)
)
valid_df = valid_df.reset_index(level=0, drop=True)

remaining_df = sorted_df.drop(index=test_df.index)
remaining_df = remaining_df.drop(index=valid_df.index)
remaining_df = (
    remaining_df.groupby("category")
    .apply(lambda x: x.sample(
        n=min(len(x), 300),
        random_state=42
    ))
    .reset_index(level=0, drop=True)
)


test_df = test_df.reset_index().drop('index', axis=1)
valid_df = valid_df.reset_index().drop('index', axis=1)
remaining_df = remaining_df.reset_index().drop('index', axis=1)

if dataset_type.lower() == "dolly":
    data_path = os.path.join("databricks-dolly-15k", str(num_clients))
elif dataset_type.lower() == "mmlu":
    data_path = os.path.join("MMLU", str(num_clients))
elif dataset_type.lower() == "glue":
    data_path = os.path.join("GLUE", str(num_clients))
else:
    raise ValueError("dataset_type must be either 'dolly' or 'truthful'")
os.makedirs(data_path, exist_ok=True)

with open(os.path.join(data_path, "global_training.json"), 'w') as outfile:
    json.dump(remaining_df.to_dict(orient='records'), outfile)

with open(os.path.join(data_path, "global_test.json"), 'w') as outfile:
    json.dump(test_df.to_dict(orient='records'), outfile)

with open(os.path.join(data_path, "global_valid.json"), 'w') as outfile:
    json.dump(valid_df.to_dict(orient='records'), outfile)

with open(os.path.join(data_path, "global_test_10.json"), 'w') as outfile:
    json.dump(test10_df.to_dict(orient='records'), outfile)

if diff_quantity:
    min_size = 0
    min_require_size = 40

    N = len(remaining_df)
    feasible = N // num_clients
    if feasible < min_require_size:
        print(f">>> [WARN] N={N} is too small for min_require_size={min_require_size}. "
              f"Setting min_require_size={feasible} instead.")
        min_require_size = feasible

    category_uniques = remaining_df["category"].unique().tolist()

    max_tries = 200 
    tries = 0

    while min_size < min_require_size and tries < max_tries:
        tries += 1
        idx_partition = [[] for _ in range(num_clients)]

        for k in range(len(category_uniques)):
            category_rows_k = remaining_df.loc[remaining_df["category"] == category_uniques[k]]
            category_rows_k_index = category_rows_k.index.values

            np.random.shuffle(category_rows_k_index)

            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_partition)]
            )
            proportions = proportions / proportions.sum()

            split_points = (np.cumsum(proportions) * len(category_rows_k_index)).astype(int)[:-1]

            idx_partition = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_partition, np.split(category_rows_k_index, split_points))
            ]

        min_size = min(len(x) for x in idx_partition)
        print(f">>> min client data size so far: {min_size} (try {tries}/{max_tries})")

    if min_size < min_require_size:
        print(f">>> [WARN] Could not reach min_require_size={min_require_size}. "
              f"Proceeding with min_size={min_size}.")
else:

    num_shards_per_clients = 2    
    remaining_df_index = remaining_df.index.values

    shards = np.array_split(remaining_df_index, int(num_shards_per_clients * num_clients))

    random.shuffle(shards)

    shards = [shards[i:i + num_shards_per_clients] for i in range(0, len(shards), num_shards_per_clients)]

    idx_partition = [np.concatenate(shards[n]).tolist() for n in range(num_clients)]

for client_id, idx in enumerate(idx_partition):
    print(
        "\n Generating the local training dataset of Client_{}".format(client_id)
    )

    sub_remaining_df = remaining_df.loc[idx]

    sub_remaining_df = sub_remaining_df.reset_index().drop('index', axis=1)
    sub_remaining_df_dic = sub_remaining_df.to_dict(orient='records')

    with open(os.path.join(data_path, "local_training_{}.json".format(client_id)), 'w') as outfile:
        json.dump(sub_remaining_df_dic, outfile)
