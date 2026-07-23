import os
import json
import argparse
from collections import defaultdict
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, LoraConfig, get_peft_model, set_peft_model_state_dict
from tqdm import tqdm
import re

if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def set_seed(seed: int):
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


def get_instruction(ex: dict) -> str:
    return str(ex.get("instruction") or ex.get("Question") or "").strip()


def get_context(ex: dict) -> str:
    return str(ex.get("context") or ex.get("input") or "").strip()


def get_response_text(ex: dict) -> str:
    return normalize_response(ex.get("response") or ex.get("Best Answer") or ex.get("answer") or "")


def get_category(ex: dict) -> str:
    return str(ex.get("category") or ex.get("Category") or "none")


def save_text(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def format_nlg_example(example: dict, include_answer: bool = True) -> str:
    instruction = get_instruction(example)
    context = get_context(example)
    response = get_response_text(example)

    prompt = f"Instruction: {instruction}\n"
    if context:
        prompt += f"Context: {context}\n"
    if include_answer and response:
        prompt += f"Response: {response}\n"
    return prompt + "\n"


def gen_prompt_nlg(examples_list, subject, shot):
    prompt = f"The following are instruction-response examples about {subject}.\n\n"
    k = min(shot, len(examples_list))
    if k <= 0:
        return prompt
    sampled = random.sample(examples_list, k)
    for ex in sampled:
        prompt += format_nlg_example(ex, include_answer=True)
    return prompt


def build_nlg_prompt(query_ex: dict, demos=None) -> str:
    prompt = ""
    if demos is not None:
        for demo in demos:
            prompt += format_nlg_example(demo, include_answer=True)
    prompt += format_nlg_example(query_ex, include_answer=False)
    prompt += "Response:"
    return prompt


def compute_answer_nll(model, tokenizer, query_ex: dict, demos=None, max_length: int = 2048) -> float:
    answer_text = get_response_text(query_ex)
    if answer_text == "":
        return float("inf")

    prompt = build_nlg_prompt(query_ex, demos=demos)
    answer = " " + answer_text
    if tokenizer.eos_token is not None:
        answer += tokenizer.eos_token

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

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        nll = float(outputs.loss.detach().cpu())
    if was_training:
        model.train()
    return nll


def generate_answer(model, tokenizer, query_ex: dict, demos=None, max_length: int = 2048, max_new_tokens: int = 128) -> str:
    prompt = build_nlg_prompt(query_ex, demos=demos)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return output


def append_db_line(db_path: str, cat: str, example_ex: dict, meta: dict = None):
    obj = {"cat": str(cat), "example": example_ex}
    if isinstance(meta, dict) and len(meta) > 0:
        obj["meta"] = meta
    with open(db_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_model_and_tokenizer(
    base_model: str,
    lora_path: str = None,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_target_modules=None,
    load_gen_model: bool = False,
):
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)

    gen_model = None
    if load_gen_model:
        print(">>> Loading generation model for proposed mode")
        gen_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
        gen_model.eval()
    else:
        print(">>> Skip generation model loading for zero/standard/metaicl mode")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if lora_path and str(lora_path).lower() not in ["none", "null", ""]:
        print(f">>> Loading LoRA from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch_dtype)
    else:
        print(">>> Using base model without LoRA")

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
    t = str(text).strip()
    if len(t) < min_chars:
        return False
    if len(t.split()) < min_words:
        return False
    return True


INSTRUCTION_PHRASES = [
    "rewrite", "follow", "format", "output", "hint", "example", "sample",
    "testcase", "do not", "don't", "must", "should", "train", "dataset",
    "kaggle", "huggingface", "github", "http://", "https://", "www.", "markdown",
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


def parse_generated_nlg(text: str, cat: str) -> dict | None:
    text = str(text).strip()
    text = text.split("### OUTPUT")[-1].strip()

    inst_match = re.search(r"Instruction:\s*(.*?)(?:\nContext:|\nResponse:|$)", text, flags=re.S | re.I)
    ctx_match = re.search(r"Context:\s*(.*?)(?:\nResponse:|$)", text, flags=re.S | re.I)
    resp_match = re.search(r"Response:\s*(.*)$", text, flags=re.S | re.I)

    if inst_match is None or resp_match is None:
        return None

    instruction = inst_match.group(1).strip()
    context = ctx_match.group(1).strip() if ctx_match is not None else ""
    response = resp_match.group(1).strip()

    if instruction == "" or response == "":
        return None

    return {
        "instruction": instruction,
        "context": context,
        "response": response,
        "category": cat,
    }


def is_valid_generated_example(generated_ex: dict, query_item: dict) -> bool:
    if generated_ex is None:
        return False

    gen_inst = get_instruction(generated_ex)
    gen_resp = get_response_text(generated_ex)
    query_inst = get_instruction(query_item)

    if not min_length_filter(gen_inst, min_chars=10, min_words=3):
        return False
    if not min_length_filter(gen_resp, min_chars=5, min_words=2):
        return False
    if not instruction_filter(gen_inst):
        return False
    if not instruction_filter(gen_resp):
        return False
    if not hard_markdown_ban(gen_inst):
        return False
    if not hard_markdown_ban(gen_resp):
        return False
    if not overlap_control(query_inst, gen_inst, threshold=0.75):
        return False
    return True


def build_generation_prompt(query_item: dict) -> str:
    seed_example = format_nlg_example(query_item, include_answer=True).strip()
    return (
        "You are generating one new instruction-response example.\n"
        "Generate a new example that can be used as an in-context demonstration.\n"
        "Do NOT copy the seed example.\n"
        "Do NOT add explanations, markdown, URLs, code, or extra sections.\n"
        "The response must be factual, concise, and directly answer the instruction.\n\n"
        "Seed example:\n"
        f"{seed_example}\n\n"
        "Output must follow EXACTLY this format:\n"
        "Instruction: ...\n"
        "Context: ...\n"
        "Response: ...\n\n"
        "### OUTPUT\n"
    )


def generate_candidate_example(base_model, tokenizer, query_item: dict, cat: str, max_length: int = 2048) -> dict | None:
    gen_prompt = build_generation_prompt(query_item)
    inputs = tokenizer(gen_prompt, return_tensors="pt", truncation=True, max_length=max_length).to(base_model.device)
    input_len = inputs.input_ids.shape[1]
    max_new = max(32, min(192, max_length - input_len))

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

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    generated_text = generated_text.split("### OUTPUT")[-1].strip()
    return parse_generated_nlg(generated_text, cat)


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


def safe_ppl(nll: float) -> float:
    if not np.isfinite(nll):
        return float("inf")
    return float(np.exp(min(nll, 20)))




def _rouge_tokens(text: str):
    text = str(text).lower().strip()
    return re.findall(r"[a-z0-9가-힣]+", text)


def _lcs_length(a, b) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = [0] * (len(short) + 1)
    for tok_l in long:
        curr = [0] * (len(short) + 1)
        for j, tok_s in enumerate(short, start=1):
            if tok_l == tok_s:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = _rouge_tokens(prediction)
    ref_tokens = _rouge_tokens(reference)
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / max(len(pred_tokens), 1)
    recall = lcs / max(len(ref_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def fmt_pct(x):
    return f"{x * 100:.2f}"

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
    do_generate: bool = False,
    max_new_tokens: int = 128,
):
    flags = eval_mode_to_flags(mode)
    use_pool = flags["use_pool"]
    use_db = flags["use_db"]
    set_seed(seed)

    online_db_added_total = 0
    online_db_added_by_task = defaultdict(int)

    local_example_blocks = defaultdict(list)
    if use_pool:
        train_path = os.path.join(train_pool_root, f"local_training_{client_id}.json")
        with open(train_path, "r", encoding="utf-8") as f:
            train_pool = json.load(f)
        for ex in train_pool:
            c = get_category(ex)
            local_example_blocks[c].append(ex)

    nll_example_blocks = defaultdict(list)
    db_path = None
    if use_db:
        db_path = os.path.join(proposed_db_root, f"example_db_client{client_id}.jsonl")
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    c = obj.get("cat", None)
                    ex_obj = obj.get("example", None)
                    if c is None or not isinstance(ex_obj, dict):
                        continue
                    nll_example_blocks[c].append(ex_obj)
        else:
            print(f"[WARNING] DB not found: {db_path} → start with empty DB")

    cat_dict = defaultdict(list)
    for item in global_test:
        cat_dict[get_category(item)].append(item)

    rows = []
    task_rougel_sum = defaultdict(float)
    task_nll_sum = defaultdict(float)
    task_count = defaultdict(int)
    all_rougels = []
    all_nlls = []

    for cat, items in cat_dict.items():
        for q_idx, query_item in enumerate(tqdm(items, desc=f"client={client_id} seed={seed} cat={cat}", leave=False)):
            k = random.randint(k_min, k_max)

            if mode == "zero":
                k = 0
                selected_examples = []

            elif mode in ["standard", "metaicl"]:
                local_examples = local_example_blocks.get(cat, [])
                k = min(k, len(local_examples))
                selected_examples = random.sample(local_examples, k) if k > 0 else []

            elif mode == "proposed":
                if base_model is None:
                    raise ValueError("proposed mode needs a generation model, but base_model is None.")
                local_examples = local_example_blocks.get(cat, [])
                db_examples = nll_example_blocks.get(cat, [])
                total_available_examples = len(local_examples) + len(db_examples)
                combined = local_examples + db_examples

                if total_available_examples >= 100:
                    selected_examples = random.sample(combined, min(k, len(combined))) if k > 0 else []
                else:
                    selected_examples = []
                    no_demo_nll = compute_answer_nll(
                        model=model,
                        tokenizer=tokenizer,
                        query_ex=query_item,
                        demos=None,
                        max_length=max_length,
                    )

                    for _ in range(k):
                        generated_ex = generate_candidate_example(
                            base_model=base_model,
                            tokenizer=tokenizer,
                            query_item=query_item,
                            cat=cat,
                            max_length=max_length,
                        )

                        if not is_valid_generated_example(generated_ex, query_item):
                            if len(local_examples) > 0:
                                selected_examples.append(random.choice(local_examples))
                            continue

                        with_demo_nll = compute_answer_nll(
                            model=model,
                            tokenizer=tokenizer,
                            query_ex=query_item,
                            demos=[generated_ex],
                            max_length=max_length,
                        )
                        nll_gain = no_demo_nll - with_demo_nll

                        if nll_gain > 0:
                            selected_examples.append(generated_ex)
                            nll_example_blocks[cat].append(generated_ex)
                            online_db_added_total += 1
                            online_db_added_by_task[cat] += 1
                        else:
                            if len(local_examples) > 0:
                                selected_examples.append(random.choice(local_examples))
            else:
                raise ValueError(f"Unknown mode: {mode}")

            reference_text = get_response_text(query_item)
            prediction_text = generate_answer(
                model=model,
                tokenizer=tokenizer,
                query_ex=query_item,
                demos=selected_examples,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
            )
            rouge_l = rouge_l_f1(prediction_text, reference_text)

            answer_nll = compute_answer_nll(
                model=model,
                tokenizer=tokenizer,
                query_ex=query_item,
                demos=selected_examples,
                max_length=max_length,
            )
            answer_ppl = safe_ppl(answer_nll)

            zero_nll = compute_answer_nll(
                model=model,
                tokenizer=tokenizer,
                query_ex=query_item,
                demos=None,
                max_length=max_length,
            )
            nll_gain_from_zero = zero_nll - answer_nll

            task_rougel_sum[cat] += float(rouge_l)
            task_nll_sum[cat] += float(answer_nll)
            task_count[cat] += 1
            all_rougels.append(float(rouge_l))
            all_nlls.append(float(answer_nll))

            row = {
                "seed": seed,
                "client_id": client_id,
                "mode": mode,
                "category": cat,
                "query_idx": q_idx,
                "instruction": get_instruction(query_item),
                "reference": reference_text,
                "prediction": prediction_text,
                "rouge_l": float(rouge_l),
                "rouge_l_pct": float(rouge_l * 100.0),
                "answer_nll": float(answer_nll),
                "answer_ppl": float(answer_ppl),
                "zero_nll": float(zero_nll),
                "zero_ppl": float(safe_ppl(zero_nll)),
                "nll_gain_from_zero": float(nll_gain_from_zero),
                "k_sampled": int(k),
                "k_selected": int(len(selected_examples)),
            }
            rows.append(row)

    overall_rougel = float(np.mean(all_rougels)) if len(all_rougels) > 0 else 0.0
    overall_nll = float(np.mean(all_nlls)) if len(all_nlls) > 0 else float("inf")
    overall_ppl = safe_ppl(overall_nll)

    per_task_rougel = {c: task_rougel_sum[c] / max(task_count[c], 1) for c in task_count.keys()}
    per_task_nll = {c: task_nll_sum[c] / max(task_count[c], 1) for c in task_count.keys()}
    per_task_ppl = {c: safe_ppl(per_task_nll[c]) for c in per_task_nll.keys()}

    online_stats = {
        "client_id": client_id,
        "seed": seed,
        "mode": mode,
        "overall_rouge_l": overall_rougel,
        "overall_rouge_l_pct": overall_rougel * 100.0,
        "overall_nll": overall_nll,
        "overall_ppl": overall_ppl,
        "per_task_rouge_l": per_task_rougel,
        "per_task_nll": per_task_nll,
        "per_task_ppl": per_task_ppl,
        "per_task_total": dict(task_count),
        "online_db_added_total": int(online_db_added_total),
        "online_db_added_by_task": {k: int(v) for k, v in online_db_added_by_task.items()},
    }

    return overall_rougel, rows, db_path, online_stats
def parse_seeds(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) < 1:
        raise ValueError("Empty --seeds")
    seeds = [int(p) for p in parts]
    if len(seeds) < 3:
        print(f"[WARNING] seeds are < 3: {seeds}")
    return seeds


def mean_std(vals):
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def fmt_float(x):
    return f"{x:.4f}"



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
    ap.add_argument("--do_generate", action="store_true", help="Kept for backward compatibility. ROUGE-L evaluation always generates predictions.")
    ap.add_argument("--max_new_tokens", type=int, default=128)

    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    os.makedirs(args.save_dir, exist_ok=True)

    flags = eval_mode_to_flags(args.mode)
    if flags["use_pool"] and args.train_pool_root is None:
        raise ValueError(f"{args.mode} needs --train_pool_root")
    if flags["use_db"] and args.proposed_db_root is None:
        raise ValueError("proposed needs --proposed_db_root")
    if args.mode == "metaicl":
        if args.metaicl_lora_root is None or str(args.metaicl_lora_root).strip() == "":
            raise ValueError("metaicl needs --metaicl_lora_root")
        if args.lora_weights_path not in [None, "", "none", "null", "None"]:
            print("[WARNING] metaicl mode: --lora_weights_path is ignored.")

    print(f">>> DEVICE: {device}")
    print(f">>> mode={args.mode} seeds={seeds}")
    print(f">>> final metric=ROUGE-L F1 (higher is better)")
    print(f">>> k_min/k_max={args.k_min}/{args.k_max}")
    print(f">>> train_pool_root={args.train_pool_root}")
    print(f">>> proposed_db_root={args.proposed_db_root}")

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
        load_gen_model=(args.mode == "proposed"),
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
    online_stats_path = os.path.join(args.save_dir, "online_stats.jsonl")
    for p in [rows_path, online_stats_path]:
        if os.path.exists(p):
            os.remove(p)

    seed_client_overall_rougel = defaultdict(dict)
    used_db_paths = defaultdict(dict)
    used_metaicl_lora_paths = defaultdict(dict)
    sc_task_rougel_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    sc_task_nll_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    sc_task_total = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    all_categories = sorted({get_category(ex) for ex in global_test})

    # zero-shot does not use client-specific local pools or DBs.
    # Evaluating it once avoids repeated identical work across clients.
    evaluated_client_ids = [0] if args.mode == "zero" else list(range(args.num_clients))
    print(f">>> evaluated_client_ids={evaluated_client_ids}")

    for seed in seeds:
        print("\n" + "=" * 90)
        print(f"[Seed {seed}] START")
        print("=" * 90)
        set_seed(seed)

        for cid in evaluated_client_ids:
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

            overall_rougel, rows, used_db_path, online_stats = evaluate_one_client_one_seed(
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
                do_generate=True,
                max_new_tokens=args.max_new_tokens,
            )

            seed_client_overall_rougel[seed][cid] = overall_rougel
            if used_db_path is not None:
                used_db_paths[str(seed)][str(cid)] = used_db_path

            with open(rows_path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    s = int(r["seed"])
                    c = int(r["client_id"])
                    t = str(r["category"])
                    sc_task_rougel_sum[s][c][t] += float(r["rouge_l"])
                    sc_task_nll_sum[s][c][t] += float(r["answer_nll"])
                    sc_task_total[s][c][t] += 1

            with open(online_stats_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(online_stats, ensure_ascii=False) + "\n")

            print(f"[Seed {seed} | Client {cid}] overall_rougeL={overall_rougel * 100:.2f} overall_nll={online_stats['overall_nll']:.4f}")

    seed_client_mean_rougel = {}
    for s in seeds:
        client_scores = list(seed_client_overall_rougel[s].values())
        seed_client_mean_rougel[s] = float(np.mean(client_scores)) if len(client_scores) > 0 else 0.0
    overall_mean_rougel, overall_std_rougel = mean_std([seed_client_mean_rougel[s] for s in seeds])

    seed_task_rougel = defaultdict(dict)
    seed_task_nll = defaultdict(dict)
    for s in seeds:
        for t in all_categories:
            client_rougel_vals = []
            client_nll_vals = []
            for cid in evaluated_client_ids:
                tot = sc_task_total[s][cid][t]
                if tot == 0:
                    continue
                client_rougel_vals.append(sc_task_rougel_sum[s][cid][t] / tot)
                client_nll_vals.append(sc_task_nll_sum[s][cid][t] / tot)
            seed_task_rougel[s][t] = float(np.mean(client_rougel_vals)) if len(client_rougel_vals) > 0 else 0.0
            seed_task_nll[s][t] = float(np.mean(client_nll_vals)) if len(client_nll_vals) > 0 else float("inf")

    seed_avg_rougel = {s: float(np.mean([seed_task_rougel[s][t] for t in all_categories])) for s in seeds}
    avg_mean_rougel, avg_std_rougel = mean_std([seed_avg_rougel[s] for s in seeds])

    task_mean_rougel, task_std_rougel = {}, {}
    task_mean_nll, task_std_nll = {}, {}
    for t in all_categories:
        m, sd = mean_std([seed_task_rougel[s][t] for s in seeds])
        task_mean_rougel[t], task_std_rougel[t] = m, sd
        mn, sdn = mean_std([seed_task_nll[s][t] for s in seeds])
        task_mean_nll[t], task_std_nll[t] = mn, sdn

    print("\n" + "#" * 90)
    print(f"FINAL | mode={args.mode} | seeds={seeds}")
    print("#" * 90)
    print(f"[Info] Overall ROUGE-L (mean over seeds of client-mean): {overall_mean_rougel * 100:.2f} ± {overall_std_rougel * 100:.2f}")
    print(f"[Table Avg] Macro mean ROUGE-L over categories: {avg_mean_rougel * 100:.2f} ± {avg_std_rougel * 100:.2f}")

    header = ["Avg_RougeL"] + [f"{t}_RougeL" for t in all_categories]
    mean_row = [fmt_pct(avg_mean_rougel)] + [fmt_pct(task_mean_rougel[t]) for t in all_categories]
    std_row = [fmt_pct(avg_std_rougel)] + [fmt_pct(task_std_rougel[t]) for t in all_categories]

    print("\n" + "=" * 90)
    print("[FINAL Dolly/NLG Table-ready Results] ROUGE-L F1, higher is better")
    print("Dolly_NLG\t" + "\t".join(header))
    print("Mean\t" + "\t".join(mean_row))
    print("Std\t" + "\t".join(std_row))

    table_tsv_path = os.path.join(args.save_dir, "dolly_rougel_table_mean_std.tsv")
    with open(table_tsv_path, "w", encoding="utf-8") as f:
        f.write("Dolly_NLG\t" + "\t".join(header) + "\n")
        f.write("Mean\t" + "\t".join(mean_row) + "\n")
        f.write("Std\t" + "\t".join(std_row) + "\n")

    table_json_path = os.path.join(args.save_dir, "dolly_rougel_table_mean_std.json")
    out = {
        "mode": args.mode,
        "seeds": seeds,
        "num_clients": args.num_clients,
        "evaluated_client_ids": evaluated_client_ids,
        "definition": {
            "rouge_l": "ROUGE-L F1 between generated prediction and gold response; values are stored in 0-1 scale, table values are multiplied by 100",
            "seed_task_rouge_l": "mean over clients of per-category ROUGE-L within a seed",
            "avg": "macro mean over categories within a seed; then mean/std over seeds",
            "note": "Higher ROUGE-L is better. NLL/PPL are kept only as diagnostic values and for proposed example curation.",
        },
        "avg_rouge_l": {"mean": avg_mean_rougel, "std": avg_std_rougel},
        "avg_rouge_l_pct": {"mean": avg_mean_rougel * 100.0, "std": avg_std_rougel * 100.0},
        "categories": {
            t: {
                "mean_rouge_l": task_mean_rougel[t],
                "std_rouge_l": task_std_rougel[t],
                "mean_rouge_l_pct": task_mean_rougel[t] * 100.0,
                "std_rouge_l_pct": task_std_rougel[t] * 100.0,
                "diagnostic_mean_nll": task_mean_nll[t],
                "diagnostic_std_nll": task_std_nll[t],
            }
            for t in all_categories
        },
        "seed_task_rouge_l": {str(s): {t: seed_task_rougel[s][t] for t in all_categories} for s in seeds},
        "seed_avg_rouge_l": {str(s): seed_avg_rougel[s] for s in seeds},
        "overall_clientmean_rouge_l": {
            "seed_client_mean_rouge_l": {str(s): seed_client_mean_rougel[s] for s in seeds},
            "mean": overall_mean_rougel,
            "std": overall_std_rougel,
        },
    }
    with open(table_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    summary = {
        "base_model": args.base_model,
        "lora_weights_path": args.lora_weights_path,
        "metaicl_lora_root": args.metaicl_lora_root,
        "global_test_path": args.global_test_path,
        "num_clients": args.num_clients,
        "evaluated_client_ids": evaluated_client_ids,
        "mode": args.mode,
        "k_min": int(args.k_min),
        "k_max": int(args.k_max),
        "seeds": seeds,
        "seed_client_overall_rouge_l": {str(k): v for k, v in seed_client_overall_rougel.items()},
        "table_avg_macro_rouge_l": {"mean": avg_mean_rougel, "std": avg_std_rougel},
        "table_categories_rouge_l": {t: {"mean": task_mean_rougel[t], "std": task_std_rougel[t]} for t in all_categories},
        "diagnostic_table_categories_nll": {t: {"mean": task_mean_nll[t], "std": task_std_nll[t]} for t in all_categories},
        "used_db_paths": used_db_paths,
        "used_metaicl_lora_paths": used_metaicl_lora_paths,
        "artifacts": {
            "rows_path": rows_path,
            "online_stats_path": online_stats_path,
            "table_tsv_path": table_tsv_path,
            "table_json_path": table_json_path,
        },
    }

    summary_path = os.path.join(args.save_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n>>> Saved:")
    print(f"- rows:         {rows_path}")
    print(f"- online_stats: {online_stats_path}")
    print(f"- table_tsv:    {table_tsv_path}")
    print(f"- table_json:   {table_json_path}")
    print(f"- summary:      {summary_path}")
if __name__ == "__main__":
    main()
