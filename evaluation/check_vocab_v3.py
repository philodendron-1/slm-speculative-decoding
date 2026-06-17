import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

TARGET_PATH  = "/data/sonnet18s/models/qwen2.5-7b"
BASE_PATH    = "/data/sonnet18s/models/qwen2.5-0.5b"
LORA_PATH_R4 = "/data/sonnet18s/lora_aligned_final/final"
LORA_PATH_DS = "/data/sonnet18s/lora_distillspec/final"
OLD_VOCAB, NEW_VOCAB = 151936, 152064
DEVICE = "cuda"

SYSTEM_PROMPT = """You are an English tutor evaluating a Korean speaker's spoken interpretation of a Korean sentence into English.

Classify the error using ONE of the following categories ONLY:
- word choice
- tense
- word order
- preposition
- naturalness
- grammar
- subject-verb agreement
- verb form
- article
- redundancy
- factual error
- omission

Respond in this exact format:
ERROR_TYPE: <category>
CORRECTION: <corrected sentence>
SHORT_REASON: <brief reason>

"""

# 7B가 다양한 카테고리를 고민할 만한 애매한 샘플들
SAMPLES = [
    {
        "ko": "저는 매일 아침 운동을 했어요.",
        "en": "I exercise every morning.",
        "desc": "tense vs naturalness 애매"
    },
    {
        "ko": "그 영화는 정말 감동적이었어요.",
        "en": "The movie was very impress.",
        "desc": "word choice vs verb form 애매"
    },
    {
        "ko": "내일 회의가 있어서 준비해야 해요.",
        "en": "Tomorrow I have meeting so I must prepare.",
        "desc": "article vs naturalness 애매"
    },
]

def expand_vocab(model):
    diff = NEW_VOCAB - OLD_VOCAB
    old_emb = model.model.embed_tokens.weight.data
    pad_emb = torch.zeros(diff, old_emb.shape[1], dtype=old_emb.dtype, device=old_emb.device)
    e = torch.nn.Embedding(NEW_VOCAB, old_emb.shape[1]).to(old_emb.device, old_emb.dtype)
    e.weight.data = torch.cat([old_emb, pad_emb], dim=0)
    model.model.embed_tokens = e
    old_lm = model.lm_head.weight.data
    pad_lm = torch.zeros(diff, old_lm.shape[1], dtype=old_lm.dtype, device=old_lm.device)
    h = torch.nn.Linear(old_lm.shape[1], NEW_VOCAB, bias=False).to(old_lm.device, old_lm.dtype)
    h.weight.data = torch.cat([old_lm, pad_lm], dim=0)
    model.lm_head = h
    model.config.vocab_size = NEW_VOCAB
    return model

def get_top5(model, inputs, tokenizer):
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
        top5 = logits.topk(5)
        probs = torch.softmax(logits, dim=-1)
        results = []
        for i, (v, idx) in enumerate(zip(top5.values, top5.indices)):
            token = tokenizer.decode([idx.item()])
            prob = probs[idx].item()
            results.append((token, prob))
            print(f"  {i+1}위: '{token}' ({prob*100:.1f}%)", flush=True)
        return results

tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH)

# 모델 로드
print("모델 로드 중...", flush=True)
target = AutoModelForCausalLM.from_pretrained(
    TARGET_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE).eval()

base = AutoModelForCausalLM.from_pretrained(
    BASE_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE)
base = expand_vocab(base)
draft_r4 = PeftModel.from_pretrained(base, LORA_PATH_R4).merge_and_unload().eval()

base2 = AutoModelForCausalLM.from_pretrained(
    BASE_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE)
base2 = expand_vocab(base2)
draft_ds = PeftModel.from_pretrained(base2, LORA_PATH_DS).merge_and_unload().eval()

for sample in SAMPLES:
    prompt = SYSTEM_PROMPT + f"""### Instruction:
Korean original: {sample['ko']}
Learner's English attempt: {sample['en']}

### Response:
ERROR_TYPE:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    print(f"\n{'='*60}", flush=True)
    print(f"샘플: {sample['desc']}", flush=True)
    print(f"한국어: {sample['ko']}", flush=True)
    print(f"통역:   {sample['en']}", flush=True)
    print(f"{'='*60}", flush=True)

    print("TARGET (7B):", flush=True)
    get_top5(target, inputs, tokenizer)

    print("DRAFT r=4 (OOD):", flush=True)
    get_top5(draft_r4, inputs, tokenizer)

    print("DistillSpec:", flush=True)
    get_top5(draft_ds, inputs, tokenizer)

print("\n완료!", flush=True)
