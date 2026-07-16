from fed_utils import client_selection, GeneralClient
from tqdm import tqdm
import fire
import os, json, random
import torch
import copy
import numpy as np
from typing import List
from datetime import datetime
from torch.nn import CrossEntropyLoss
import matplotlib
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from collections import defaultdict
import torch.backends.cudnn as cudnn

def set_seed(seed):
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
    return str(resp).strip()

def map_label_for_cat(cat: str, lbl: str) -> str:
    if str(cat).strip().lower() == "mnli":
        m = {
            "Entailment": "A",
            "Contradiction": "B",
            "Neutral": "C",
        }
        return m.get(str(lbl).strip(), str(lbl).strip())
    return str(lbl).strip()

def safe_last_token_id(tokenizer, text: str) -> int:
    if text is None:
        return -1
    t = str(text).strip()
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

def append_db_line(db_path: str, cat: str, example_ex: dict, meta: dict = None):
    obj = {"cat": str(cat), "example": example_ex}
    if isinstance(meta, dict) and len(meta) > 0:
        obj["meta"] = meta
    with open(db_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        
def consensus_fedavg(lora_dirs):
    def load_lora_state(lora_dir):
        return torch.load(
            os.path.join(lora_dir, "pytorch_model.bin"),
            map_location="cpu"
        )

    ref_state = load_lora_state(lora_dirs[0])
    keys = sorted(ref_state.keys())

    def lora_state_to_vector(state_dict):
        return torch.cat([state_dict[k].flatten() for k in keys])

    def vector_to_lora_state(vec):
        new_sd = {}
        idx = 0
        for k in keys:
            numel = ref_state[k].numel()
            new_sd[k] = vec[idx: idx + numel].reshape(ref_state[k].shape)
            idx += numel
        return new_sd

    lora_states = [load_lora_state(d) for d in lora_dirs]
    task_vecs = torch.stack(
        [lora_state_to_vector(sd) for sd in lora_states]
    )  

    n, d = task_vecs.shape

    avg_vec = task_vecs.mean(dim=0)

    sign_sum = torch.sign(task_vecs).sum(dim=0)
    consensus_strength = sign_sum.abs() / n  

    merged_vec = avg_vec * consensus_strength

    merged_state = vector_to_lora_state(merged_vec)
    return merged_state


def validate_glue(model, tokenizer, valid_data, max_length=2048):
    model.eval()
    correct, losses = [], []
    ce_loss = CrossEntropyLoss()

    label_space = {}
    for ex in valid_data:
        cat = ex["category"]
        lbl = normalize_response(ex["response"])
        lbl = map_label_for_cat(cat, lbl)
        label_space.setdefault(cat, [])
        if lbl not in label_space[cat]:
            label_space[cat].append(lbl)

    label_token_ids = {}
    for cat, labels in label_space.items():
        ids = []
        for lbl in labels:
            lid = safe_last_token_id(tokenizer, lbl)
            assert_label_ok(lid, tokenizer, context=f"validate_glue cat={cat} lbl={lbl}")
            ids.append(lid)
        label_token_ids[cat] = ids

    for ex in valid_data:
        cat = ex["category"]
        labels = label_space[cat]
        label_ids = label_token_ids[cat]

        prompt = ex["instruction"].rstrip() + "\nAnswer: "
        gt_label = normalize_response(ex["response"])
        gt_label = map_label_for_cat(cat, gt_label)

        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        ).input_ids.to(model.device)

        with torch.no_grad():
            logit = model(input_ids=input_ids).logits[0, -1]

        logits = torch.stack([logit[lid] for lid in label_ids])

        pred_idx = torch.argmax(torch.softmax(logits, dim=0)).item()
        pred_label = labels[pred_idx]
        correct.append(pred_label == gt_label)

        gt_idx = labels.index(gt_label)
        loss = ce_loss(logits.unsqueeze(0), torch.tensor([gt_idx]).to(model.device))
        losses.append(loss.item())

    return float(np.mean(correct)), float(np.mean(losses))


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
    if cat in ["qqp", "mrpc", "rte", "qnli", "mnli"]:
        if ("setence1:" not in t) and ("sentence1:" not in t):
            return False
        if ("setence2:" not in t) and ("sentence2:" not in t):
            return False
        if "question:" not in t:
            return False
        return True
    if cat in ["cola", "sst2"]:
        if ("setence:" not in t) and ("sentence:" not in t):
            return False
        if "question:" not in t:
            return False
        return True
    return True
import re
def overlap_ratio(a: str, b: str) -> float:
    A = set(re.findall(r"[a-zA-Z]+", a.lower()))
    B = set(re.findall(r"[a-zA-Z]+", b.lower()))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)
def overlap_control(orig: str, generated: str, threshold: float = 0.75) -> bool:
    return overlap_ratio(orig, generated) < threshold

def glue_metaicl(
        global_model: str = "",
        data_path: str = "./data",
        output_dir: str = "./lora-model_fl_icl/",
        shot: int = 0,
        client_selection_strategy: str = "random",
        client_selection_frac: float = 1,
        num_communication_rounds: int = 1,
        num_clients: int = 5,
        local_batch_size: int = 8,
        local_micro_batch_size: int = 2,
        local_num_epochs: int = 5,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
        train_on_inputs: bool = True,
        group_by_length: bool = False,
        ExampleGen: bool = False,
        use_bf16: bool = True,
        allow_tf32: bool = True,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"shot: {shot}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"use_bf16: {use_bf16}\n"
            f"allow_tf32: {allow_tf32}\n"
        )

    set_seed(1234)

    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    data_path = os.path.join(data_path, str(num_clients))
    assert os.path.exists(data_path), "Please generate the data files for each client"

    global_valid_path = os.path.join(data_path, "global_valid.json")
    from datasets import load_dataset
    global_valid = load_dataset("json", data_files=global_valid_path)["train"]
    
    def build_label_space(dataset):
        label_space = defaultdict(set)
        for ex in dataset:
            cat = ex["category"]
            lbl = normalize_response(ex["response"])
            lbl = map_label_for_cat(cat, lbl)
            label_space[cat].add(lbl)
        return {cat: sorted(list(labels)) for cat, labels in label_space.items()}

    LABEL_SPACE = build_label_space(global_valid)

    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.bfloat16 if (use_bf16 and device == "cuda") else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        global_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        global_model,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    output_dir = os.path.join(output_dir, str(num_clients))
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        model.config.use_cache = False
    except:
        pass
    try:
        model.config.attn_implementation = "sdpa"
    except:
        pass

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config).to(device)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        
    base_model = AutoModelForCausalLM.from_pretrained(
        global_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    previously_selected_clients_set = set()
    local_dataset_len_dict = dict()

    LABEL_ID_CACHE = {}
    def get_label_id(cat, lbl):
        key = (cat, lbl)
        if key in LABEL_ID_CACHE:
            return LABEL_ID_CACHE[key]
        lid = safe_last_token_id(tokenizer, lbl)
        assert_label_ok(lid, tokenizer, context=f"get_label_id cat={cat} lbl={lbl}")
        LABEL_ID_CACHE[key] = lid
        return lid
    
    def preprocess_GivenExample_wrapper(ex, all_examples, round=None, client_id=None, output_dir=None):
        cat = ex["category"]
        same_cat = [t for t in all_examples if t.get("category", None) == cat]

        k_total = random.randint(1, 3)
        k = min(k_total, len(same_cat))

        examples = gen_prompt_json(same_cat, cat, k)

        query = format_example_json(ex, include_answer=False)
        prompt = examples + query

        os.makedirs(output_dir, exist_ok=True)
        debug_file = os.path.join(output_dir, f"round{round}_client{client_id}.txt")
        save_text(debug_file, "=====================================")
        save_text(debug_file, prompt + "\n")

        tokenized = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False)

        gt = normalize_response(ex["response"])
        gt = map_label_for_cat(cat, gt)
        label_id = get_label_id(cat, gt)

        ids = tokenized["input_ids"]
        tokenized["labels"] = [-100] * (len(ids) - 1) + [label_id]
        return tokenized

    def preprocess_ExampleGen_wrapper(ex, all_examples, round=None, client_id=None, output_dir=None):
        cat = ex["category"]
        same_cat = [t for t in all_examples if t.get("category", None) == cat] 
        local_example_blocks = same_cat

        k = random.randint(1, 3)
        
        query = format_example_json(ex, include_answer=False)

        db_path = os.path.join(output_dir, f"example_db_client{client_id}.jsonl")
        mi_example_blocks = []
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if obj.get("cat") != cat:
                        continue
                    ex_obj = obj.get("example", None)
                    if isinstance(ex_obj, dict) and ex_obj.get("category") == cat:
                        mi_example_blocks.append(ex_obj)
        total_available = len(local_example_blocks) + len(mi_example_blocks)

        label_space = LABEL_SPACE[cat]
        label_token_ids = []
        for lbl in label_space:
            lbl2 = map_label_for_cat(cat, lbl)
            lid = safe_last_token_id(tokenizer, lbl2)
            assert_label_ok(lid, tokenizer, context=f"preprocess cat={cat} lbl={lbl2}")
            label_token_ids.append(lid)
        gt = normalize_response(ex["response"])
        gt = map_label_for_cat(cat, gt)
        gt_label_index = label_space.index(gt)  
        label_id = get_label_id(cat, gt)
        
        if total_available >= 100:
            combined = local_example_blocks + mi_example_blocks
            selected_examples = random.sample(combined, k)
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
                    "You are generating a new natural language example.\n"
                    "Rewrite ONLY the content after each field key.\n"
                    "Keep all field keys EXACTLY the same.\n"
                    "Do NOT add instructions, explanations, rules, code, URLs, or extra sections.\n"
                    "Each field must contain a complete natural sentence.\n"
                    "Preserve the Question line exactly as given.\n\n"
                    "Output must follow EXACTLY this structure:\n\n"
                    f"{query.strip()}\n\n"
                    "### OUTPUT\n"
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
                generated_text = generated_text.split("### OUTPUT")[-1].strip() 
                
                if not min_length_filter(generated_text) or not instruction_filter(generated_text) or not hard_markdown_ban(generated_text) or not enforce_task_structure(cat, generated_text) or not overlap_control(query, generated_text, threshold=0.75):
                    if len(local_example_blocks) > 0:
                        selected_examples.append(random.choice(local_example_blocks))
                else:
                    generated_ex = {
                        "instruction": generated_text,
                        "context": "",
                        "response": "",  
                        "category": cat
                    }
                    
                    examples = gen_prompt_json([generated_ex], cat, 1)
                    gen_prompt = examples + query

                    inputs2 = tokenizer(gen_prompt, return_tensors="pt").to(device)

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
                    
                    pred_index = int(np.argmax(new_probs_np)) 
                    correct_prediction = (pred_index == gt_label_index)

                    if (mi > 0) and correct_prediction:
                        append_db_line(
                            db_path,
                            cat=cat,
                            example_ex=generated_ex,
                            meta={"mi_entropy_gain": float(mi)},
                        )
                        selected_examples.append(generated_ex)
                    else:
                        if len(local_example_blocks) > 0:
                            selected_examples.append(random.choice(local_example_blocks))
        
        examples = gen_prompt_json(selected_examples, cat, k)
        prompt = examples + query

        os.makedirs(output_dir, exist_ok=True)
        debug_file = os.path.join(output_dir, f"round{round}_client{client_id}.txt")
        save_text(debug_file, "=====================================")
        save_text(debug_file, prompt + "\n")

        tokenized = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False)

        ids = tokenized["input_ids"]
        tokenized["labels"] = [-100] * (len(ids) - 1) + [label_id]
        return tokenized


    for round in tqdm(range(num_communication_rounds)):
        
        print("\nConducting the client selection")
        selected_clients_set = [0, 1, 2, 3, 4]

        for client_id in selected_clients_set:
            client = GeneralClient(client_id, model, data_path, output_dir)

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))

            client.preprare_local_dataset(preprocess_ExampleGen_wrapper, round)

            client.local_val_set_size = local_val_set_size

            client.build_local_trainer(
                tokenizer,
                local_micro_batch_size,
                gradient_accumulation_steps,
                local_num_epochs,
                local_learning_rate,
                group_by_length,
            )

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()

            print("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = (
                client.terminate_local_training(
                    round, local_dataset_len_dict, previously_selected_clients_set
                )
            )
            del client

        print("Collecting the weights of clients and performing aggregation")

        client_lora_dirs = [
            os.path.join(output_dir, str(round), f"local_output_{cid}")
            for cid in selected_clients_set
        ]

        global_state = consensus_fedavg(
            client_lora_dirs
        )
        
        set_peft_model_state_dict(model, global_state, "default")

        adapter_path = os.path.join(output_dir, str(round), f"adapter_model_round{round}.bin")
        torch.save(global_state, adapter_path)
        adapter_path2 = os.path.join(output_dir, f"adapter_model.bin")
        torch.save(global_state, adapter_path2)
        print(f"[Round {round}] Saved global LoRA adapter → {adapter_path2}")
        
        if "Qwen/Qwen3-4B" in global_model:
            model_tag= "qwen4b"
        elif "Qwen/Qwen3-14B" in global_model:
            model_tag= "qwen14b"
        elif "meta-llama/Llama-3.2-3B" in global_model:
            model_tag= "llama3b"
        elif "meta-llama/Llama-3.1-8B" in global_model:
            model_tag= "llama8b"

        model.peft_config["default"].save_pretrained(output_dir)

        acc, vloss = validate_glue(model, tokenizer, global_valid)
        msg = f"[Round {round}] VALID acc={acc:.4f} loss={vloss:.4f}"
        print(msg)
        save_text(os.path.join(output_dir, "train_log.txt"), msg)


if __name__ == "__main__":
    fire.Fire(
        {
            "icl_fl": glue_metaicl,
        }
    )
