#!/usr/bin/env python3
# Diagnostic: show exact token ids/tokens generated and different decodes.
import os, time, traceback
CONDA_BASE = os.popen("conda info --base").read().strip()
ENV_PYTHON = os.path.join(CONDA_BASE, "envs", "biomni", "bin", "python")
print("Using ENV_PYTHON:", ENV_PYTHON)

prompt = "Hello, how are you? I want you to respond with a short sentence."
max_new_tokens = 32
temperature = 0.7
do_sample = True
top_p = 0.9

print("Prompt:", prompt)
try:
    from scripts.run_local_infer import LocalInfer
    import torch
except Exception as e:
    print("Import error:", e)
    traceback.print_exc()
    raise SystemExit(1)

infer = LocalInfer(model_dir=os.getenv("LOCAL_SFT_MODEL_DIR", "/root/autodl-tmp/Biomni-main/models/Qwen2.5-7B-Instruct/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"))
tok = infer.tokenizer
model = infer.model

inputs = tok(prompt, return_tensors="pt", truncation=True)
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask
print("input_ids shape:", tuple(input_ids.shape))
print("input token ids (first row):", input_ids[0].tolist())
print("decoded input (skip_special_tokens=False):", tok.decode(input_ids[0], skip_special_tokens=False))

gen_kwargs = dict(
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    do_sample=do_sample,
    eos_token_id=tok.eos_token_id if getattr(tok, "eos_token_id", None) is not None else None,
    pad_token_id=tok.pad_token_id if getattr(tok, "pad_token_id", None) is not None else None,
)
print("gen_kwargs (as used):", gen_kwargs)

# Generate with current kwargs
with torch.no_grad():
    out = model.generate(input_ids=input_ids.to(getattr(model, "device", "cpu")),
                         attention_mask=attention_mask.to(getattr(model, "device", "cpu")),
                         **gen_kwargs)

print("out type:", type(out))
try:
    print("out.shape:", tuple(out.shape))
except Exception:
    pass

# Extract generated part
input_len = input_ids.shape[-1]
generated_ids = out[0][input_len:]
print("generated_ids (raw):", generated_ids.tolist() if hasattr(generated_ids, "tolist") else repr(generated_ids))

# Show tokens for generated_ids
try:
    gen_tokens = tok.convert_ids_to_tokens(generated_ids.tolist() if hasattr(generated_ids, "tolist") else [int(x) for x in generated_ids])
    print("generated tokens (convert_ids_to_tokens):", gen_tokens)
except Exception as e:
    print("convert_ids_to_tokens failed:", e)

# Decode generated with skip_special_tokens=True and False
try:
    dec_skip = tok.decode(generated_ids, skip_special_tokens=True)
    dec_noskip = tok.decode(generated_ids, skip_special_tokens=False)
    full_dec_skip = tok.decode(out[0], skip_special_tokens=True)
    full_dec_noskip = tok.decode(out[0], skip_special_tokens=False)
    print("decoded generated (skip_special_tokens=True):", repr(dec_skip))
    print("decoded generated (skip_special_tokens=False):", repr(dec_noskip))
    print("decoded full out[0] (skip_special_tokens=True):", repr(full_dec_skip)[:400])
    print("decoded full out[0] (skip_special_tokens=False):", repr(full_dec_noskip)[:400])
except Exception as e:
    print("decode error:", e)
    traceback.print_exc()

# Try generate with eos_token_id=None to see if model emits real tokens
try:
    gen_kwargs2 = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=do_sample, eos_token_id=None)
    print("Trying generate with eos_token_id=None ...")
    with torch.no_grad():
        out2 = model.generate(input_ids=input_ids.to(getattr(model, "device", "cpu")),
                              attention_mask=attention_mask.to(getattr(model, "device", "cpu")),
                              **gen_kwargs2)
    print("out2.shape:", tuple(out2.shape))
    generated_ids2 = out2[0][input_len:]
    print("generated_ids2:", generated_ids2.tolist())
    print("tokens2:", tok.convert_ids_to_tokens(generated_ids2.tolist()))
    print("decoded2 skip_special_tokens=True:", repr(tok.decode(generated_ids2, skip_special_tokens=True)))
    print("decoded2 skip_special_tokens=False:", repr(tok.decode(generated_ids2, skip_special_tokens=False)))
except Exception as e:
    print("generate with eos_token_id=None failed:", e)
    traceback.print_exc()

print("done.")