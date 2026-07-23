import os
import json
import argparse
from collections import defaultdict
import random
import numpy as np
import torch.nn.functional as F
import torch
import shutil
from pathlib import Path
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, LoraConfig, get_peft_model, set_peft_model_state_dict
import re  
try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None
if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
task2category = {
    "abstract_algebra": "STEM",
    "anatomy": "STEM",
    "astronomy": "STEM",
    "business_ethics": "Social Sciences",
    "clinical_knowledge": "STEM",
    "college_biology": "STEM",
    "college_chemistry": "STEM",
    "college_computer_science": "STEM",
    "college_mathematics": "STEM",
    "college_medicine": "STEM",
    "college_physics": "STEM",
    "computer_security": "STEM",
    "conceptual_physics": "STEM",
    "econometrics": "Social Sciences",
    "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM",
    "formal_logic": "STEM",
    "global_facts": "Other",
    "high_school_biology": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM",
    "high_school_european_history": "Humanities",
    "high_school_geography": "Social Sciences",
    "high_school_government_and_politics": "Humanities",
    "high_school_macroeconomics": "Social Sciences",
    "high_school_mathematics": "STEM",
    "high_school_microeconomics": "Social Sciences",
    "high_school_physics": "STEM",
    "high_school_psychology": "Social Sciences",
    "high_school_statistics": "STEM",
    "high_school_us_history": "Humanities",
    "high_school_world_history": "Humanities",
    "human_aging": "Other",
    "human_sexuality": "Social Sciences",
    "international_law": "Humanities",
    "jurisprudence": "Humanities",
    "logical_fallacies": "STEM",
    "machine_learning": "STEM",
    "management": "Social Sciences",
    "marketing": "Social Sciences",
    "medical_genetics": "STEM",
    "miscellaneous": "Other",
    "moral_disputes": "Humanities",
    "moral_scenarios": "Humanities",
    "nutrition": "STEM",
    "philosophy": "Humanities",
    "prehistory": "Humanities",
    "professional_accounting": "STEM",
    "professional_law": "Humanities",
    "professional_medicine": "STEM",
    "professional_psychology": "STEM",
    "public_relations": "Social Sciences",
    "security_studies": "Social Sciences",
    "sociology": "Social Sciences",
    "us_foreign_policy": "Social Sciences",
    "virology": "STEM",
    "world_religions": "Humanities",
}
    
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def normalize_response(resp):
    if isinstance(resp, dict):
        for k in ["label", "answer", "response", "target"]:
            if k in resp:
                resp = resp[k]
                break
    if isinstance(resp, (list, tuple)):
        if len(resp) == 0:
            return ""
        resp = resp[0]
    s = str(resp).strip()

    m = re.search(r"\b([ABCD])\b", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return s

def map_label_for_cat(cat: str, lbl: str) -> str:
    return str(lbl).strip()


def safe_last_token_id(tokenizer, text: str) -> int:
    t = str(text).strip() if text is not None else ""
    if t == "":
        return -1
    ids = tokenizer.encode(" " + t, add_special_tokens=False)
    if len(ids) == 0:
        ids = tokenizer.encode(t, add_special_tokens=False)
    if len(ids) == 0:
        return -1
    return int(ids[-1])

def assert_label_ok(label_id: int, tokenizer, context: str = ""):
    vsz = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if label_id < 0 or label_id >= vsz:
        raise ValueError(
            f"[BAD_LABEL_ID] label_id={label_id} vocab_size={vsz} context={context}"
        )

def save_text(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def format_example_json(example, include_answer=True):
    prompt = str(example["instruction"]).rstrip()

    if include_answer:
        cat = example.get("category", "")
        label = normalize_response(example.get("response", ""))
        label = map_label_for_cat(cat, label)
        if label != "": 
            prompt += "\nAnswer: " + label
    return prompt + "\n\n"

def gen_prompt_json(examples_list, subject, shot):
    prompt = (
        f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    )
    k = min(shot, len(examples_list))
    if k <= 0:
        return prompt
    sampled = random.sample(examples_list, k)
    for ex in sampled:
        prompt += format_example_json(ex, include_answer=True)
    return prompt
    
def load_model_and_tokenizer(
    base_model: str,
    lora_path: str = None,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_target_modules=None,
):
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    
    gen_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    gen_model.eval()
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if lora_path:
        print(f">>> Loading LoRA (dir) from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16)
    else:
        print("pre-trained model")

    model = model.to(device)
    model.eval()
    return model, gen_model, tokenizer


def eval_mode_to_flags(mode: str):
    mode = mode.lower().strip()
    if mode not in ["zero", "standard", "metaicl", "proposed"]:
        raise ValueError(f"Unknown mode: {mode}")
    if mode == "zero":
        return {"use_pool": False, "use_db": False}
    if mode in ["standard", "metaicl"]:
        return {"use_pool": True, "use_db": False}
    return {"use_pool": True, "use_db": True}


def min_length_filter(text: str, min_chars: int = 20, min_words: int = 5) -> bool:
    t = text.strip()
    if len(t) < min_chars:
        return False
    if len(t.split()) < min_words:
        return False
    return True
INSTRUCTION_PHRASES = [
    "rewrite", "follow", "instruction", "format", "output",
    "hint", "example", "sample", "testcase",
    "do not", "don't", "must", "should",
    "input", "output", "train", "dataset",
    "kaggle", "huggingface", "github",
    "http://", "https://", "www."
]
def instruction_filter(text: str) -> bool:
    low = text.lower()
    for phrase in INSTRUCTION_PHRASES:
        if phrase in low:
            return False
    return True
def hard_markdown_ban(text: str) -> bool:
    t = text.lower()
    if "```" in t:
        return False
    if t.startswith("#"):
        return False
    if "markdown" in t:
        return False
    if "```bash" in t or "```python" in t:
        return False
    return True
def enforce_task_structure(cat: str, text: str) -> bool:
    t = text.lower()
    has_q = "question:" in t
    has_a = ("a:" in t) or ("a." in t)
    has_b = ("b:" in t) or ("b." in t)
    has_c = ("c:" in t) or ("c." in t)
    has_d = ("d:" in t) or ("d." in t)
    has_ans = "answer:" in t
    return has_q and has_a and has_b and has_c and has_d and has_ans
def overlap_ratio(a: str, b: str) -> float:
    A = set(re.findall(r"[a-zA-Z]+", a.lower()))
    B = set(re.findall(r"[a-zA-Z]+", b.lower()))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)
def overlap_control(orig: str, generated: str, threshold: float = 0.75) -> bool:
    return overlap_ratio(orig, generated) < threshold

def ensure_lora_wrapped_model(model, lora_r: int, lora_alpha: int, lora_dropout: float, lora_target_modules):
    if isinstance(model, PeftModel):
        return model
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


def evaluate_one_client_one_seed(
    model,
    base_model,
    tokenizer,
    global_test: list,
    client_id: int,
    seed: int,
    mode: str,
    k_min: int,
    k_max: int,
    train_pool_root: str,
    proposed_db_root: str,
    max_length: int = 2048,
):
    flags = eval_mode_to_flags(mode)
    use_pool = flags["use_pool"]
    use_db = flags["use_db"]
    
    set_seed(seed)
    
    LABEL_SPACE = defaultdict(lambda: ["A","B","C","D"])
    
    online_db_added_total = 0
    online_db_added_by_task = defaultdict(int)

    local_example_blocks = defaultdict(list)  
    if use_pool:
        train_path = os.path.join(train_pool_root, f"local_training_{client_id}.json")
        with open(train_path, "r", encoding="utf-8") as f:
            train_pool = json.load(f)  
        for ex in train_pool:
            c = ex.get("category", "none")
            local_example_blocks[c].append(ex) 

    mi_example_blocks = defaultdict(list) 
    db_path = None
    if use_db:
        db_path = os.path.join(proposed_db_root, f"example_db_client{client_id}.jsonl")
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    c = obj.get("cat", None)
                    ex_obj = obj.get("example", None)
                    if c is None:
                        continue
                    if isinstance(ex_obj, dict):
                        mi_example_blocks[c].append(ex_obj)
        else:
            print(f"[WARNING] DB not found: {db_path} → start with empty DB")
                
    cat_dict = defaultdict(list)
    for item in global_test:
        cat = item.get("category", "none")
        cat_dict[cat].append(item)

    all_cors = []
    rows = []
    task_correct = defaultdict(int)
    task_total = defaultdict(int)

    for cat, items in cat_dict.items():
        label_space = LABEL_SPACE[cat] 
        label_token_ids = []
        for lbl in label_space:
            lbl2 = map_label_for_cat(cat, lbl)
            lid = safe_last_token_id(tokenizer, lbl2)
            assert_label_ok(lid, tokenizer, context=f"preprocess cat={cat} lbl={lbl2}")
            label_token_ids.append(lid)

        for q_idx, query_item in enumerate(items):

            k = random.randint(k_min, k_max)
            query = format_example_json(query_item, include_answer=False)
            if mode == "zero":
                k = 0
                selected_examples = []
            elif mode in ["standard", "metaicl"]:
                local_examples = local_example_blocks.get(cat, [])
                k = min(k, len(local_examples))
                if k > 0:
                    selected_examples = random.sample(local_examples, k)
                else:
                    selected_examples = []
            elif mode == "proposed":
                local_examples = local_example_blocks.get(cat, [])
                mi_examples = mi_example_blocks.get(cat, [])
                total_available_examples = len(local_examples) + len(mi_examples)
                combined = local_examples + mi_examples

                if total_available_examples >= 50:
                    if k > 0:
                        selected_examples = random.sample(combined, k)
                    else:
                        selected_examples = []

                else:
                    selected_examples = []
                    inputs = tokenizer(query, return_tensors="pt").to(device)

                    with torch.no_grad():
                        logits = model(**inputs).logits[0, -1]
                        
                    label_logits = torch.stack([logits[lid] for lid in label_token_ids])
                    base_probs = torch.softmax(label_logits, dim=0)
                    base_probs_np = base_probs.detach().cpu().numpy()
                    base_probs_np /= (np.sum(base_probs_np) + 1e-12)

                    current_entropy = -np.sum(
                        base_probs_np * np.log(base_probs_np + 1e-12)
                    )

                    for _ in range(k):
                        gen_prompt = (
                            "Create ONE new multiple-choice question similar in topic and difficulty.\n"
                            "Output must follow EXACTLY this format (6 lines minimum):\n"
                            "Question: <one question>\n"
                            "A: <choice A>\n"
                            "B: <choice B>\n"
                            "C: <choice C>\n"
                            "D: <choice D>\n"
                            "Answer: <A/B/C/D>\n"
                            "No extra text. No explanations. No URLs.\n\n"
                            "Here is an example question (do NOT copy it):\n"
                            f"{query_item['instruction'].rstrip()}\n"
                            "\nBEGIN_EXAMPLE\n"
                        )
                        inputs = tokenizer(gen_prompt, return_tensors="pt").to(device)
                        input_len = inputs.input_ids.shape[1]
                        max_length=2048
                        max_new = max(16, min(128, max_length - input_len))

                        with torch.no_grad():
                            generated_ids = base_model.generate(
                                **inputs,
                                max_new_tokens=max_new,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.90,
                                repetition_penalty=1.12,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.eos_token_id,
                            )

                        generated_text = tokenizer.decode(
                            generated_ids[0], skip_special_tokens=True
                        ).strip()
                        generated_text = generated_text.split("BEGIN_EXAMPLE")[-1].strip() 
                        
                        if not min_length_filter(generated_text) or not instruction_filter(generated_text) or not hard_markdown_ban(generated_text) or not enforce_task_structure(cat, generated_text) or not overlap_control(query, generated_text, threshold=0.75):
                            if len(local_examples) > 0:
                                selected_examples.append(random.choice(local_examples))
                        else:
      
                            generated_ex = {
                                "instruction": generated_text,
                                "context": "",
                                "response": "",     
                                "category": cat
                            }
                            
                            examples = gen_prompt_json([generated_ex], cat, 1)
                            gen_prompt = examples + query
                            
                            inputs2 = tokenizer(gen_prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)

                            with torch.no_grad():
                                logits2 = model(**inputs2).logits[0, -1]

                            label_logits2 = torch.stack(
                                [logits2[lid] for lid in label_token_ids]
                            )

                            new_probs = torch.softmax(label_logits2, dim=0)
                            new_probs_np = new_probs.detach().cpu().numpy()
                            new_probs_np /= (np.sum(new_probs_np) + 1e-12)

                            new_entropy = -np.sum(
                                new_probs_np * np.log(new_probs_np + 1e-12)
                            )

                            mi = current_entropy - new_entropy
                            
                            if mi > 0: 
                                selected_examples.append(generated_ex)
                                mi_example_blocks[cat].append(generated_ex)
                                online_db_added_total += 1
                                online_db_added_by_task[cat] += 1
                            else: 
                                if len(local_examples) > 0:
                                    selected_examples.append(random.choice(local_examples))

            examples = gen_prompt_json(selected_examples, cat, k)
            prompt = examples + query

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs.input_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids=input_ids).logits[0, -1]

            label_logits = torch.stack([logits[lid] for lid in label_token_ids])
            probs = torch.softmax(label_logits, dim=0).detach().cpu().numpy()

            pred_idx = int(np.argmax(probs)) 
            pred_label = label_space[pred_idx]
            ref_label = normalize_response(query_item["response"])
            ref_label = map_label_for_cat(cat, ref_label)
            cor = (pred_label == ref_label)
            
            task_total[cat] += 1
            task_correct[cat] += int(cor)
            all_cors.append(cor)

            row = {
                "seed": seed,
                "client_id": client_id,
                "mode": mode,
                "category": cat,
                "query_idx": q_idx,
                "reference": ref_label,
                "prediction": pred_label,
                "correct": int(cor),
                "k_sampled": int(k),
                "k_selected": int(len(selected_examples)),
            }
            rows.append(row)

    overall_acc = float(np.mean(all_cors)) if len(all_cors) > 0 else 0.0
    per_task_acc = {c: task_correct[c] / max(task_total[c], 1) for c in task_total.keys()}

    online_stats = {
        "client_id": client_id,
        "seed": seed,
        "mode": mode,
        "overall_acc": overall_acc,
        "per_task_acc": per_task_acc,
        "per_task_total": dict(task_total),
        "online_db_added_total": int(online_db_added_total),
        "online_db_added_by_task": {k: int(v) for k, v in online_db_added_by_task.items()}
    }

    return overall_acc, rows, db_path, online_stats

def parse_seeds(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) < 1:
        raise ValueError("Empty --seeds")
    seeds = [int(p) for p in parts]
    if len(seeds) < 3:
        print(f"[WARNING] seeds are < 3: {seeds}")
    return seeds


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--lora_weights_path", type=str, default=None) 

    ap.add_argument("--global_test_path", type=str, required=True)
    ap.add_argument("--num_clients", type=int, required=True)

    ap.add_argument("--train_pool_root", type=str, default=None)     
    ap.add_argument("--proposed_db_root", type=str, default=None)    

    ap.add_argument("--mode", type=str, required=True, choices=["zero", "standard", "metaicl", "proposed"])

    ap.add_argument("--k_min", type=int, default=0)
    ap.add_argument("--k_max", type=int, default=3)

    ap.add_argument("--seeds", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])

    ap.add_argument("--max_length", type=int, default=2048)

    ap.add_argument("--metaicl_lora_root", type=str, default=None)

    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    os.makedirs(args.save_dir, exist_ok=True)

    flags = eval_mode_to_flags(args.mode)
    
    if flags["use_pool"]:
        if args.train_pool_root is None:
            raise ValueError(f"{args.mode} needs --train_pool_root (e.g., .../GLUE/5)")
    if flags["use_db"]:
        if args.proposed_db_root is None:
            raise ValueError("proposed needs --proposed_db_root")
    if args.mode == "metaicl":
        if args.metaicl_lora_root is None or str(args.metaicl_lora_root).strip() == "":
            raise ValueError("metaicl needs --metaicl_lora_root")
        if args.lora_weights_path not in [None, "", "none", "null", "None"]:
            print("[WARNING] metaicl mode: --lora_weights_path is ignored (client LoRAs are loaded per-client).")

    print(f">>> DEVICE: {device}")
    print(f">>> mode={args.mode} seeds={seeds}")
    print(f">>> k_min/k_max={args.k_min}/{args.k_max}")
    print(f">>> train_pool_root={args.train_pool_root}")
    print(f">>> proposed_db_root={args.proposed_db_root}")
    if args.mode == "metaicl":
        print(f">>> metaicl_lora_root={args.metaicl_lora_root}")

    with open(args.global_test_path, "r", encoding="utf-8") as f:
        global_test = json.load(f)
    print(f">>> Loaded global_test: {args.global_test_path} (N={len(global_test)})")

    model, gen_model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        lora_path=(None if args.mode == "metaicl" else args.lora_weights_path),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )
    

    if args.mode == "metaicl":
        model = ensure_lora_wrapped_model(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
        ).to(device)
        model.eval()

    rows_path = os.path.join(args.save_dir, "eval_rows.jsonl")
    if os.path.exists(rows_path):
        os.remove(rows_path)

    online_stats_path = os.path.join(args.save_dir, "online_stats.jsonl")
    if os.path.exists(online_stats_path):
        os.remove(online_stats_path)

    seed_client_overall = defaultdict(dict)
    used_db_paths = defaultdict(dict)
    used_metaicl_lora_paths = defaultdict(dict)

    sc_task_correct = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    sc_task_total   = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    seed_client_overall = defaultdict(dict)

    for seed in seeds:
        print("\n" + "=" * 90)
        print(f"[Seed {seed}] START")
        print("=" * 90)
        set_seed(seed)

        for cid in range(args.num_clients):

            if args.mode == "metaicl":
                lora_bin = os.path.join(
                    args.metaicl_lora_root,
                    str(args.num_clients),
                    "0",

                    f"local_output_{cid}",
                    "pytorch_model.bin",
                )
                print(f">>> [metaicl] apply client LoRA | seed={seed} cid={cid} | {lora_bin}")
                lora_sd = torch.load(lora_bin, map_location="cpu")
                set_peft_model_state_dict(model, lora_sd, "default")
                model.eval()
                used_metaicl_lora_paths[str(seed)][str(cid)] = lora_bin

            overall_acc, rows, used_db_path, online_stats = evaluate_one_client_one_seed(
                model=model,
                base_model=gen_model,
                tokenizer=tokenizer,
                global_test=global_test,
                client_id=cid,
                seed=seed,
                mode=args.mode,
                k_min=args.k_min,
                k_max=args.k_max,
                train_pool_root=args.train_pool_root,
                proposed_db_root=args.proposed_db_root,
                max_length=args.max_length,
            )

            seed_client_overall[seed][cid] = overall_acc
            if used_db_path is not None:
                used_db_paths[str(seed)][str(cid)] = used_db_path

            with open(rows_path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

                    s = int(r["seed"])
                    c = int(r["client_id"])
                    t = str(r["category"])
                    sc_task_correct[s][c][t] += int(r["correct"])
                    sc_task_total[s][c][t]   += 1

            with open(online_stats_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(online_stats, ensure_ascii=False) + "\n")

            print(f"[Seed {seed} | Client {cid}] overall_acc={overall_acc:.4f}")


    def mean_std(vals):
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    def fmt_pct(x):  
        return f"{x*100:.2f}"

    all_tasks = sorted(list(task2category.keys()))  

    seed_task_acc = defaultdict(dict)  
    for s in seeds:
        for t in all_tasks:
            client_vals = []
            for cid in range(args.num_clients):
                tot = sc_task_total[s][cid][t]
                if tot == 0:
                    continue
                client_vals.append(sc_task_correct[s][cid][t] / tot)
            seed_task_acc[s][t] = float(np.mean(client_vals)) if len(client_vals) > 0 else 0.0

    def norm_supercat(x: str) -> str:
        x = (x or "").strip()
        if x.lower() in ["humanities", "hum", "hum."]:
            return "Hum."
        if x.lower() in ["social sciences", "social science", "soc", "soc.sci", "soc sci"]:
            return "Soc.Sci"
        if x.lower() == "stem":
            return "STEM"
        return "Other"

    cat2tasks = defaultdict(list)
    for t in all_tasks:
        supercat = norm_supercat(task2category.get(t, "Other"))
        cat2tasks[supercat].append(t)

    ORDERED_SUPERCATS = ["STEM", "Hum.", "Soc.Sci", "Other"]

    seed_cat_score = defaultdict(dict)  
    seed_avg = {}                       

    for s in seeds:
        seed_avg[s] = float(np.mean([seed_task_acc[s][t] for t in all_tasks])) if len(all_tasks) > 0 else 0.0

        for C in ORDERED_SUPERCATS:
            ts = cat2tasks.get(C, [])
            seed_cat_score[s][C] = float(np.mean([seed_task_acc[s][t] for t in ts])) if len(ts) > 0 else 0.0

    avg_mean, avg_std = mean_std([seed_avg[s] for s in seeds])

    cat_mean, cat_std = {}, {}
    for C in ORDERED_SUPERCATS:
        cat_mean[C], cat_std[C] = mean_std([seed_cat_score[s][C] for s in seeds])

    header = ["Avg"] + ORDERED_SUPERCATS
    mean_row = [fmt_pct(avg_mean)] + [fmt_pct(cat_mean[C]) for C in ORDERED_SUPERCATS]
    std_row  = [fmt_pct(avg_std)]  + [fmt_pct(cat_std[C])  for C in ORDERED_SUPERCATS]

    print("\n" + "#" * 90)
    print(f"FINAL | mode={args.mode} | seeds={seeds}")
    print("#" * 90)
    print("\n" + "=" * 90)
    print("[FINAL MMLU Table-ready Results]")
    print("MMLU\t" + "\t".join(header))
    print("Mean\t" + "\t".join(mean_row))
    print("Std\t"  + "\t".join(std_row))

    table_tsv_path = os.path.join(args.save_dir, "mmlu_table_mean_std.tsv")
    with open(table_tsv_path, "w", encoding="utf-8") as f:
        f.write("MMLU\t" + "\t".join(header) + "\n")
        f.write("Mean\t" + "\t".join(mean_row) + "\n")
        f.write("Std\t"  + "\t".join(std_row) + "\n")

    table_json_path = os.path.join(args.save_dir, "mmlu_table_mean_std.json")
    out = {
        "mode": args.mode,
        "seeds": seeds,
        "num_clients": args.num_clients,
        "definition": {
            "seed_task_acc": "A_{s,t} = mean over clients of task accuracy for task t within seed s",
            "category_macro": "Category_s[C] = macro mean over tasks in category C within seed s",
            "avg_macro": "Avg_s = macro mean over ALL 57 tasks within seed s; table reports mean/std over seeds",
        },
        "avg": {"mean": avg_mean, "std": avg_std},
        "supercats": {C: {"mean": cat_mean[C], "std": cat_std[C], "num_tasks": len(cat2tasks.get(C, []))} for C in ORDERED_SUPERCATS},

        "task2category": task2category,
        "seed_task_acc": {str(s): {t: seed_task_acc[s][t] for t in all_tasks} for s in seeds},
        "seed_avg": {str(s): seed_avg[s] for s in seeds},
        "seed_supercat": {str(s): {C: seed_cat_score[s][C] for C in ORDERED_SUPERCATS} for s in seeds},
    }
    with open(table_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    latex_path = os.path.join(args.save_dir, "mmlu_table_row.tex")
    def pm(m, s):
        return f"{m*100:.2f}$_{{\\pm {s*100:.2f}}}$"
    latex_cells = [pm(avg_mean, avg_std)] + [pm(cat_mean[C], cat_std[C]) for C in ORDERED_SUPERCATS]
    latex_row = args.mode + " & " + " & ".join(latex_cells) + " \\\\"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_row + "\n")

    print("\n>>> Saved:")
    print(f"- table_tsv:  {table_tsv_path}")
    print(f"- table_json: {table_json_path}")
    print(f"- latex_row:  {latex_path}")


if __name__ == "__main__":
    main()
