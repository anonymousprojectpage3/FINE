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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_response(resp):
    if isinstance(resp, dict):
        for k in ["label", "answer", "response", "target", "Best Answer"]:
            if k in resp:
                resp = resp[k]
                break
    if isinstance(resp, (list, tuple)):
        if len(resp) == 0:
            return ""
        resp = resp[0]
    return str(resp).strip()


def get_instruction(ex):
    return str(
        ex.get("instruction")
        or ex.get("Question")
        or ex.get("question")
        or ex.get("prompt")
        or ""
    ).strip()


def get_response_text(ex):
    return str(
        ex.get("response")
        or ex.get("Best Answer")
        or ex.get("answer")
        or ex.get("target")
        or ""
    ).strip()


def get_context_text(ex):
    return str(ex.get("context") or ex.get("input") or "").strip()


def get_category(ex):
    return str(
        ex.get("category")
        or ex.get("Category")
        or ex.get("task")
        or "default"
    ).strip()


def save_text(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def append_db_line(db_path: str, cat: str, example_ex: dict, meta: dict = None):
    obj = {"cat": str(cat), "example": example_ex}
    if isinstance(meta, dict) and len(meta) > 0:
        obj["meta"] = meta
    with open(db_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def format_nlg_example(example, include_answer=True):
    instruction = get_instruction(example)
    context = get_context_text(example)
    response = get_response_text(example)

    prompt = f"Instruction: {instruction}\n"
    if context:
        prompt += f"Context: {context}\n"
    if include_answer:
        prompt += f"Response: {response}\n"
    return prompt + "\n"


def gen_prompt_nlg(examples_list, subject, shot):
    prompt = (
        "The following are truthful and helpful instruction-response examples"
        f" about {subject}.\n\n"
    )
    k = min(shot, len(examples_list))
    if k <= 0:
        return prompt
    sampled = random.sample(examples_list, k)
    for ex in sampled:
        if get_instruction(ex) and get_response_text(ex):
            prompt += format_nlg_example(ex, include_answer=True)
    return prompt


def build_nlg_query_prompt(query_ex):
    instruction = get_instruction(query_ex)
    context = get_context_text(query_ex)

    prompt = f"Instruction: {instruction}\n"
    if context:
        prompt += f"Context: {context}\n"
    prompt += "Response:"
    return prompt


def build_nlg_prompt(query_ex, demos=None):
    cat = get_category(query_ex)
    prompt = ""
    if demos is not None and len(demos) > 0:
        prompt += gen_prompt_nlg(demos, cat, len(demos))
    prompt += build_nlg_query_prompt(query_ex)
    return prompt


def parse_generated_qa(text, cat):
    text = str(text).strip()
    text = re.sub(r"</?s>", "", text).strip()

    patterns = [
        (r"Instruction:\s*(.*?)(?:\n\s*Response:|$)", r"Response:\s*(.*)"),
        (r"Question:\s*(.*?)(?:\n\s*Answer:|$)", r"Answer:\s*(.*)"),
    ]

    for q_pat, a_pat in patterns:
        q_match = re.search(q_pat, text, flags=re.S | re.I)
        a_match = re.search(a_pat, text, flags=re.S | re.I)
        if q_match is None or a_match is None:
            continue

        instruction = q_match.group(1).strip()
        response = a_match.group(1).strip()

        response = re.split(
            r"\n\s*(Instruction:|Question:|###|Context:)",
            response,
            maxsplit=1,
            flags=re.I,
        )[0].strip()

        if instruction and response:
            return {
                "instruction": instruction,
                "context": "",
                "response": response,
                "category": cat,
            }

    return None


def compute_answer_nll(model, tokenizer, query_ex, demos=None, max_length=512):
    answer_text = get_response_text(query_ex)
    if answer_text == "":
        return float("inf")

    prompt = build_nlg_prompt(query_ex, demos=demos)
    answer = " " + answer_text

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    if len(answer_ids) == 0:
        return float("inf")

    if len(answer_ids) >= max_length:
        answer_ids = answer_ids[: max_length - 1]

    max_prompt_len = max_length - len(answer_ids)
    if max_prompt_len <= 0:
        return float("inf")
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + answer_ids
    prompt_len = len(prompt_ids)

    input_ids = torch.tensor([input_ids], dtype=torch.long).to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        nll = float(outputs.loss.detach().cpu())
    if was_training:
        model.train()

    return nll


def tokenize_nlg_sft(tokenizer, prompt, answer_text, max_length=512):
    answer = " " + str(answer_text).strip()
    if tokenizer.eos_token is not None:
        answer += tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    if len(answer_ids) == 0:
        answer_ids = [tokenizer.eos_token_id]

    if len(answer_ids) >= max_length:
        answer_ids = answer_ids[: max_length - 1]

    max_prompt_len = max_length - len(answer_ids)
    if max_prompt_len <= 0:
        max_prompt_len = 1
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def validate_nlg_ppl(model, tokenizer, valid_data, max_length=512, max_samples=200):
    model.eval()
    nlls = []

    data = list(valid_data)
    if max_samples is not None and max_samples > 0 and len(data) > max_samples:
        data = random.sample(data, max_samples)

    for ex in data:
        if get_instruction(ex) == "" or get_response_text(ex) == "":
            continue
        nll = compute_answer_nll(
            model=model,
            tokenizer=tokenizer,
            query_ex=ex,
            demos=None,
            max_length=max_length,
        )
        if np.isfinite(nll):
            nlls.append(nll)

    if len(nlls) == 0:
        return float("nan"), float("nan")

    avg_nll = float(np.mean(nlls))
    avg_ppl = float(np.exp(min(avg_nll, 20)))
    return avg_nll, avg_ppl

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


def min_length_filter(text: str, min_chars: int = 20, min_words: int = 5) -> bool:
    t = str(text).strip()
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
    low = str(text).lower()
    for phrase in INSTRUCTION_PHRASES:
        if phrase in low:
            return False
    return True


def hard_markdown_ban(text: str) -> bool:
    t = str(text).lower().strip()
    if "```" in t:
        return False
    if t.startswith("#"):
        return False
    if "markdown" in t:
        return False
    if "```bash" in t or "```python" in t:
        return False
    return True


def overlap_ratio(a: str, b: str) -> float:
    A = set(re.findall(r"[a-zA-Z]+", str(a).lower()))
    B = set(re.findall(r"[a-zA-Z]+", str(b).lower()))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def overlap_control(orig: str, generated: str, threshold: float = 0.75) -> bool:
    return overlap_ratio(orig, generated) < threshold


def valid_generated_example(generated_ex, original_ex):
    if generated_ex is None:
        return False

    gen_instruction = get_instruction(generated_ex)
    gen_response = get_response_text(generated_ex)
    orig_instruction = get_instruction(original_ex)

    if not min_length_filter(gen_instruction, min_chars=10, min_words=3):
        return False
    if not min_length_filter(gen_response, min_chars=5, min_words=2):
        return False
    if not instruction_filter(gen_instruction):
        return False
    if not instruction_filter(gen_response):
        return False
    if not hard_markdown_ban(gen_instruction):
        return False
    if not hard_markdown_ban(gen_response):
        return False
    if not overlap_control(orig_instruction, gen_instruction, threshold=0.75):
        return False
    return True


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
        ExampleGen: bool = True,
        use_bf16: bool = True,
        allow_tf32: bool = True,
        valid_max_samples: int = 200,
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
            f"ExampleGen: {ExampleGen}\n"
            f"use_bf16: {use_bf16}\n"
            f"allow_tf32: {allow_tf32}\n"
            f"valid_max_samples: {valid_max_samples}\n"
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

    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if (use_bf16 and device == "cuda") else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        global_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
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
    except Exception:
        pass
    try:
        model.config.attn_implementation = "sdpa"
    except Exception:
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


    previously_selected_clients_set = set()
    local_dataset_len_dict = dict()

    def preprocess_GivenExample_wrapper(ex, all_examples, round=None, client_id=None, output_dir=None):
        cat = get_category(ex)
        same_cat = [t for t in all_examples if get_category(t) == cat and get_response_text(t) != ""]

        k_total = random.randint(1, 3)
        k = min(k_total, len(same_cat))
        selected_examples = random.sample(same_cat, k) if k > 0 else []

        prompt = gen_prompt_nlg(selected_examples, cat, k) + build_nlg_query_prompt(ex)

        os.makedirs(output_dir, exist_ok=True)
        debug_file = os.path.join(output_dir, f"round{round}_client{client_id}.txt")
        save_text(debug_file, "=====================================")
        save_text(debug_file, prompt + "\n")

        return tokenize_nlg_sft(
            tokenizer=tokenizer,
            prompt=prompt,
            answer_text=get_response_text(ex),
            max_length=cutoff_len,
        )

    def preprocess_ExampleGen_wrapper(ex, all_examples, round=None, client_id=None, output_dir=None):
        cat = get_category(ex)
        ex["category"] = cat

        same_cat = [
            t for t in all_examples
            if get_category(t) == cat and get_instruction(t) != "" and get_response_text(t) != ""
        ]
        local_example_blocks = same_cat
        k = random.randint(1, 3)

        db_path = os.path.join(output_dir, f"example_db_client{client_id}.jsonl")
        mi_example_blocks = []
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("cat") != cat:
                        continue
                    ex_obj = obj.get("example", None)
                    if (
                        isinstance(ex_obj, dict)
                        and get_category(ex_obj) == cat
                        and get_instruction(ex_obj) != ""
                        and get_response_text(ex_obj) != ""
                    ):
                        mi_example_blocks.append(ex_obj)

        total_available = len(local_example_blocks) + len(mi_example_blocks)

        if total_available >= 100:
            combined = local_example_blocks + mi_example_blocks
            selected_examples = random.sample(combined, min(k, len(combined)))
        else:
            selected_examples = []
            no_demo_nll = compute_answer_nll(
                model=model,
                tokenizer=tokenizer,
                query_ex=ex,
                demos=None,
                max_length=cutoff_len,
            )

            for _ in range(k):
                seed_example = format_nlg_example(ex, include_answer=True)
                gen_prompt = (
                    "You are generating one new truthful instruction-response example.\n"
                    "Generate a NEW example that can be used as an in-context demonstration.\n"
                    "The new example should be in the same general domain as the seed example,\n"
                    "but it must not copy the seed instruction.\n"
                    "The response must be factual, concise, and directly answer the instruction.\n"
                    "Do not add explanations, markdown, code, URLs, or extra sections.\n\n"
                    "Seed example:\n"
                    f"{seed_example}\n"
                    "Output must follow EXACTLY this format:\n"
                    "Instruction: ...\n"
                    "Response: ...\n\n"
                    "### OUTPUT\n"
                )

                inputs = tokenizer(gen_prompt, return_tensors="pt").to(device)
                input_len = inputs.input_ids.shape[1]
                max_gen_len = 2048
                max_new = max(16, min(128, max_gen_len - input_len))

                was_training = model.training
                model.eval()
                try:
                    with torch.no_grad():
                        if hasattr(model, "disable_adapter"):
                            with model.disable_adapter():
                                generated_ids = model.generate(
                                    **inputs,
                                    max_new_tokens=max_new,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_p=0.90,
                                    repetition_penalty=1.12,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.eos_token_id,
                                )
                        else:
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=max_new,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.90,
                                repetition_penalty=1.12,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                finally:
                    if was_training:
                        model.train()

                generated_text = tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                ).strip()
                generated_text = generated_text.split("### OUTPUT")[-1].strip()
                generated_ex = parse_generated_qa(generated_text, cat)

                if not valid_generated_example(generated_ex, ex):
                    if len(local_example_blocks) > 0:
                        selected_examples.append(random.choice(local_example_blocks))
                    continue

                with_demo_nll = compute_answer_nll(
                    model=model,
                    tokenizer=tokenizer,
                    query_ex=ex,
                    demos=[generated_ex],
                    max_length=cutoff_len,
                )
                nll_gain = no_demo_nll - with_demo_nll

                if np.isfinite(nll_gain) and nll_gain > 0:
                    append_db_line(
                        db_path,
                        cat=cat,
                        example_ex=generated_ex,
                        meta={
                            "nll_gain": float(nll_gain),
                            "no_demo_nll": float(no_demo_nll),
                            "with_demo_nll": float(with_demo_nll),
                            "no_demo_ppl": float(np.exp(min(no_demo_nll, 20))),
                            "with_demo_ppl": float(np.exp(min(with_demo_nll, 20))),
                        },
                    )
                    selected_examples.append(generated_ex)
                else:
                    if len(local_example_blocks) > 0:
                        selected_examples.append(random.choice(local_example_blocks))

        prompt = gen_prompt_nlg(selected_examples, cat, k) + build_nlg_query_prompt(ex)

        os.makedirs(output_dir, exist_ok=True)
        debug_file = os.path.join(output_dir, f"round{round}_client{client_id}.txt")
        save_text(debug_file, "=====================================")
        save_text(debug_file, prompt + "\n")

        return tokenize_nlg_sft(
            tokenizer=tokenizer,
            prompt=prompt,
            answer_text=get_response_text(ex),
            max_length=cutoff_len,
        )

    preprocess_fn = preprocess_ExampleGen_wrapper if ExampleGen else preprocess_GivenExample_wrapper

    for round in tqdm(range(num_communication_rounds)):
        print("\nConducting the client selection")
        selected_clients_set = list(range(num_clients))

        for client_id in selected_clients_set:
            client = GeneralClient(client_id, model, data_path, output_dir)

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset(preprocess_fn, round)
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

        global_state = consensus_fedavg(client_lora_dirs)
        set_peft_model_state_dict(model, global_state, "default")

        adapter_path = os.path.join(output_dir, str(round), f"adapter_model_round{round}.bin")
        torch.save(global_state, adapter_path)
        adapter_path2 = os.path.join(output_dir, f"adapter_model.bin")
        torch.save(global_state, adapter_path2)
        print(f"[Round {round}] Saved global LoRA adapter → {adapter_path2}")

        model.peft_config["default"].save_pretrained(output_dir)

        valid_nll, valid_ppl = validate_nlg_ppl(
            model=model,
            tokenizer=tokenizer,
            valid_data=global_valid,
            max_length=cutoff_len,
            max_samples=valid_max_samples,
        )
        msg = f"[Round {round}] VALID answer_nll={valid_nll:.4f} answer_ppl={valid_ppl:.4f}"
        print(msg)
        save_text(os.path.join(output_dir, "train_log.txt"), msg)


if __name__ == "__main__":
    fire.Fire(
        {
            "icl_fl": glue_metaicl,
        }
    )
