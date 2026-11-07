import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import argparse
import random
import json
import os

try:
    # optional dependency: only needed when running as a script
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    # optional dependency: progress bar
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    tqdm = None

@dataclass
class BeamState:
    input_ids: torch.LongTensor          # [1, t]
    past_key_values: Optional[Tuple]      # transformers cache
    cum_logprob: float                   # sum log p
    counts: torch.Tensor                 # [n_layers, n_experts] long or float
    sigma: float                         # scalar, computed from counts
    # cached artifacts from the most recent forward of this beam (used to extend selected candidates)
    last_pkv: Optional[Tuple] = None
    step_counts: Optional[torch.Tensor] = None  # [n_layers, n_experts] float (CPU)

class GateHook:
    """
    Capture gate logits for the *current forward call*.
    We assume gate/router ultimately produces logits over experts (last dim = num_experts).
    Note: some models (e.g. GPT-OSS) may implement router as a custom module (not nn.Linear)
    while still exposing parameters like `router.weight` / `router.bias`.
    """
    def __init__(self):
        # gate_slot -> logits tensor, shape [B, S, E] or [B, E] depending on implementation
        # NOTE: this is meant to store outputs for ONLY the most recent forward call; call clear() before forward.
        self.gate_logits: Dict[int, torch.Tensor] = {}

    def clear(self):
        self.gate_logits.clear()

    def make_hook(self, gate_slot: int):
        def _hook(module: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor):
            # output could be:
            # - Tensor: logits over experts
            # - tuple/list/dict: try to pick the first Tensor-like payload
            out = output
            if torch.is_tensor(out):
                self.gate_logits[gate_slot] = out.detach()
                return
            if isinstance(out, (tuple, list)):
                for x in out:
                    if torch.is_tensor(x):
                        self.gate_logits[gate_slot] = x.detach()
                        return
            if isinstance(out, dict):
                for x in out.values():
                    if torch.is_tensor(x):
                        self.gate_logits[gate_slot] = x.detach()
                        return
        return _hook

def _get_module_weight(module: nn.Module) -> Optional[torch.Tensor]:
    # Prefer a direct attribute first (common), otherwise fall back to named_parameters.
    w = getattr(module, "weight", None)
    if torch.is_tensor(w):
        return w
    try:
        for n, p in module.named_parameters(recurse=False):
            if n == "weight" and torch.is_tensor(p):
                return p
    except Exception:
        pass
    return None

def _is_router_like_module(name: str, module: nn.Module, num_experts: int) -> bool:
    lname = name.lower()
    if not (("gate" in lname) or ("router" in lname) or ("routing" in lname)):
        return False
    # Case 1: classic MoE gate is a Linear(hidden -> E)
    if isinstance(module, nn.Linear):
        try:
            return int(module.out_features) == int(num_experts)
        except Exception:
            return False
    # Case 2: custom router module that exposes `weight` with shape [E, hidden]
    w = _get_module_weight(module)
    if w is None:
        return False
    if w.dim() >= 2 and int(w.shape[0]) == int(num_experts):
        return True
    return False

def find_moe_gates(model: nn.Module, num_experts: int) -> List[Tuple[int, nn.Module, str]]:
    """
    Heuristic: find gate/router modules that produce logits over experts.
    Supports:
    - nn.Linear with out_features == num_experts (Qwen/Deepseek/Mixtral-like)
    - custom router modules exposing `weight` shaped [num_experts, hidden] (GPT-OSS-like)
    Returns list of (layer_idx, linear_module, full_name).
    layer_idx is inferred by parsing name; if not possible, use incremental index.
    """
    gates = []
    fallback_idx = 0
    for name, m in model.named_modules():
        if not _is_router_like_module(name, m, num_experts=num_experts):
            continue
        # try to parse layer index from name like "model.layers.12.mlp.router"
        layer_idx = None
        parts = name.split(".")
        for i in range(len(parts) - 1):
            if parts[i] in ("layers", "layer", "h", "blocks"):
                try:
                    layer_idx = int(parts[i + 1])
                    break
                except Exception:
                    pass
        if layer_idx is None:
            layer_idx = fallback_idx
            fallback_idx += 1
        gates.append((layer_idx, m, name))
    # sort by layer_idx for consistent ordering
    gates.sort(key=lambda x: x[0])
    return gates

def get_model_device(model, fallback: str = "cuda") -> torch.device:
    """
    Best-effort to find a device that accepts input_ids.
    If the model is sharded (accelerate device_map), parameters may live on different devices; we pick the first.
    """
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device(fallback)

def _parse_hf_device(v) -> Optional[torch.device]:
    """
    Parse accelerate hf_device_map values which could be int (gpu id), "cuda:0", "cpu", "disk", etc.
    Returns torch.device for cuda/cpu, or None for disk/unknown.
    """
    try:
        if isinstance(v, int):
            return torch.device(f"cuda:{v}")
        s = str(v)
        if s.startswith("cuda"):
            return torch.device(s)
        if s == "cpu":
            return torch.device("cpu")
        # ignore "disk" / other offload targets
        return None
    except Exception:
        return None

def get_input_device(model, fallback: str = "cuda") -> torch.device:
    """
    Best-effort: pick the device where input_ids should live.
    For sharded models (hf_device_map), try common embedding module keys; otherwise fallback to first param device.
    """
    dm = getattr(model, "hf_device_map", None)
    if isinstance(dm, dict) and len(dm) > 0:
        preferred_suffixes = (
            "model.embed_tokens",
            "model.tok_embeddings",
            "transformer.wte",
            "gpt_neox.embed_in",
            "embed_tokens",
            "wte",
        )
        # exact or suffix match
        for suf in preferred_suffixes:
            for k, v in dm.items():
                if k == suf or k.endswith(suf):
                    d = _parse_hf_device(v)
                    if d is not None:
                        return d
        # fallback: first parseable device in map
        for v in dm.values():
            d = _parse_hf_device(v)
            if d is not None:
                return d
    return get_model_device(model, fallback=fallback)

def _infer_counts_device(model, gates: List[Tuple[int, nn.Module, str]], fallback: torch.device) -> torch.device:
    """
    Decide where to keep small routing statistics (counts/step_counts).
    - If all gate modules live on the same cuda device, keep counts on that device.
    - Otherwise (multi-device / cpu / offload), keep counts on CPU to avoid cross-device transfers.
    """
    devs = []
    for _, m, _ in gates:
        try:
            w = _get_module_weight(m)
            if w is not None:
                devs.append(w.device)
        except Exception:
            pass
    uniq = {str(d) for d in devs if d is not None}
    if len(uniq) == 1:
        d0 = devs[0]
        if d0 is not None and d0.type == "cuda":
            return d0
    # sharded or offloaded: safest on CPU
    dm = getattr(model, "hf_device_map", None)
    if isinstance(dm, dict) and len(dm) > 0:
        return torch.device("cpu")
    # single-device model but couldn't infer: use fallback
    return fallback

def _repeat_past_key_values(model, pkv, repeat: int):
    """
    Repeat cache along batch dimension from 1 -> repeat.
    Supports both:
    - new transformers Cache objects (preferred): pkv.batch_repeat_interleave(repeat)
    - legacy tuple/list caches: repeat tensors on batch dim
    Returns repeated cache, or None if pkv is None.
    """
    if pkv is None:
        return None
    # new cache API (transformers>=4.35+): DynamicCache / StaticCache etc.
    if hasattr(pkv, "batch_repeat_interleave"):
        return pkv.batch_repeat_interleave(repeat)
    # some models may expose helper on model
    if hasattr(model, "_reorder_cache"):
        # cannot reorder from batch=1 to batch>1; fall through
        pass
    # legacy tuple/list of per-layer tuples of tensors
    if isinstance(pkv, (tuple, list)):
        out_layers = []
        for layer in pkv:
            if not isinstance(layer, (tuple, list)):
                out_layers.append(layer)
                continue
            new_layer = []
            for t in layer:
                if torch.is_tensor(t):
                    if t.dim() >= 1 and t.size(0) == 1:
                        new_layer.append(t.repeat_interleave(repeat, dim=0))
                    elif t.dim() >= 2 and t.size(1) == 1:
                        new_layer.append(t.repeat_interleave(repeat, dim=1))
                    else:
                        raise RuntimeError(f"Cannot repeat pkv tensor with shape {tuple(t.shape)}")
                else:
                    new_layer.append(t)
            out_layers.append(tuple(new_layer))
        return tuple(out_layers)
    raise RuntimeError(f"Unknown past_key_values type for repeat: {type(pkv)}")

def _select_past_key_values(model, pkv, indices: torch.LongTensor, batch_size: int):
    """
    Select/reorder cache rows by indices (supports duplicates).
    Supports:
    - model._reorder_cache(pkv, indices) if present (works for both new Cache and legacy tuples)
    - pkv.reorder_cache(indices) for new Cache objects
    - legacy tuple/list caches: index_select on batch dim
    """
    if pkv is None:
        return None
    # Many HF models implement this specifically for beam search.
    if hasattr(model, "_reorder_cache"):
        try:
            return model._reorder_cache(pkv, indices)
        except Exception:
            pass
    if hasattr(pkv, "reorder_cache"):
        return pkv.reorder_cache(indices)
    if isinstance(pkv, (tuple, list)):
        out_layers = []
        for layer in pkv:
            if not isinstance(layer, (tuple, list)):
                out_layers.append(layer)
                continue
            new_layer = []
            for t in layer:
                if torch.is_tensor(t):
                    idx = indices.to(t.device)
                    if t.dim() >= 1 and t.size(0) == batch_size:
                        new_layer.append(t.index_select(0, idx))
                    elif t.dim() >= 2 and t.size(1) == batch_size:
                        new_layer.append(t.index_select(1, idx))
                    else:
                        raise RuntimeError(
                            f"Cannot select pkv tensor with shape {tuple(t.shape)} for batch_size={batch_size}"
                        )
                else:
                    new_layer.append(t)
            out_layers.append(tuple(new_layer))
        return tuple(out_layers)
    raise RuntimeError(f"Unknown past_key_values type for select: {type(pkv)}")

def infer_top_k(model) -> int:
    # config has highest priority
    if hasattr(model, "config") and hasattr(model.config, "num_experts_per_tok"):
        try:
            return max(1, int(model.config.num_experts_per_tok))
        except Exception:
            pass
    # try layer0 for common HF MoE architectures (Qwen/Deepseek/Mixtral-like)
    try:
        layer0 = model.model.layers[0]
        if hasattr(layer0, "mlp"):
            mlp = layer0.mlp
            for attr in ("top_k", "num_experts_per_tok"):
                if hasattr(mlp, attr):
                    return max(1, int(getattr(mlp, attr)))
        if hasattr(layer0, "block_sparse_moe"):
            bsm = layer0.block_sparse_moe
            for attr in ("top_k", "num_experts_per_tok"):
                if hasattr(bsm, attr):
                    return max(1, int(getattr(bsm, attr)))
    except Exception:
        pass
    raise ValueError("Failed to infer MoE top_k (num_experts_per_tok). Please specify it explicitly.")

def _sample_from_topk_logprobs(logprobs: torch.Tensor, k: int) -> int:
    """
    logprobs: [1, V]
    Returns a sampled token id (python int) from top-k distribution.
    """
    k = min(int(k), logprobs.size(-1))
    topv, topi = torch.topk(logprobs, k=k, dim=-1)  # [1, k], [1, k]
    probs = torch.exp(topv)  # [1, k]
    probs = probs / probs.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(probs, num_samples=1).item()  # in [0, k)
    return int(topi[0, idx].item())

def compute_sigma_from_counts_batched(counts: torch.Tensor) -> torch.Tensor:
    """
    counts: [B, L, E] float
    Returns sigma: [B] float, averaged over layers.
    """
    if counts.dim() != 3:
        raise ValueError(f"counts must be [B, L, E], got {tuple(counts.shape)}")
    B, L, E = counts.shape
    if L == 0 or E == 0:
        return torch.zeros((B,), device=counts.device, dtype=torch.float32)
    tot = counts.sum(dim=-1, keepdim=True)  # [B, L, 1]
    # avoid div-by-zero; we'll mask later
    u = counts / (tot + 1e-12)  # [B, L, E]
    mean = u.mean(dim=-1, keepdim=True)  # [B, L, 1]
    var = ((u - mean) ** 2).sum(dim=-1) / max(E - 1, 1)  # [B, L]
    sigma_l = torch.sqrt(var)  # [B, L]
    mask = (tot.squeeze(-1) > 0)  # [B, L]
    sigma_l = torch.where(mask, sigma_l, torch.zeros_like(sigma_l))
    return sigma_l.mean(dim=-1)  # [B]

def _step_counts_from_gate_logits(
    gate_logits: Dict[int, torch.Tensor],
    gate_slot_to_layer_compact: List[int],
    L: int,
    num_experts: int,
    top_k: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build per-beam step routing counts from captured gate logits.
    Returns: [B, L, E] on `device`.
    Note: Top-k on logits is equivalent to top-k on softmax probabilities (monotonic).
    """
    step_counts = torch.zeros((batch_size, L, num_experts), device=device, dtype=dtype)
    if len(gate_logits) == 0:
        return step_counts
    kk = min(int(top_k), int(num_experts))
    for gate_slot, gl in gate_logits.items():
        l = gate_slot_to_layer_compact[gate_slot]
        # common shapes:
        # - [B, S, E]
        # - [B, E]
        # - [B*S, E] (some implementations flatten tokens)
        if gl.dim() == 3:
            gl_last = gl[:, -1, :]  # [B, E]
        elif gl.dim() == 2:
            if gl.size(0) == batch_size:
                gl_last = gl  # [B, E]
            elif gl.size(0) % batch_size == 0:
                # flatten tokens: [B*S, E] -> [B, S, E] -> take last token
                s = gl.size(0) // batch_size
                gl_last = gl.view(batch_size, s, num_experts)[:, -1, :]  # [B, E]
            else:
                raise RuntimeError(
                    f"Gate logits batch mismatch: expected {batch_size} or multiple, got {gl.size(0)}"
                )
        else:
            raise RuntimeError(f"Unexpected gate logits dim={gl.dim()} shape={tuple(gl.shape)}")
        idx = torch.topk(gl_last, k=kk, dim=-1).indices  # [B, kk]
        idx = idx.to(device)
        ones = torch.ones_like(idx, device=device, dtype=step_counts.dtype)
        step_counts[:, l, :].scatter_add_(dim=1, index=idx, src=ones)
    return step_counts

def compute_sigma_from_counts(counts: torch.Tensor) -> float:
    """
    counts: [L, E] (float)
    For each layer, compute usage frequency and its stddev across experts, then average across layers.
    """
    L, E = counts.shape
    sigmas = []
    for l in range(L):
        c = counts[l]
        tot = float(c.sum().item())
        if tot <= 0:
            sigmas.append(0.0)
            continue
        u = c / tot  # frequency
        mean = u.mean()
        # unbiased std like paper's 1/(E-1) sum (u-mean)^2
        var = ((u - mean) ** 2).sum() / max(E - 1, 1)
        sigmas.append(float(torch.sqrt(var).item()))
    return float(sum(sigmas) / max(L, 1))

@torch.no_grad()
def ebss_generate_calib(
    model,
    tokenizer,
    num_seqs: int = 128,
    seqlen: int = 512,
    w: int = 4,
    tau: float = 1.2,
    temperature: float = 1.0,
    vocab_topk: int = 50,
    device: str = "cuda",
    warmup_tokens: int = 3,
    stop_on_eos: bool = False,
    progress: bool = True,
    show_token_progress: bool = False,
):
    """
    EBSS-style self sampling with probability-guided pruning.
    Keeps w beams; at each step expands each beam with vocab_topk candidates and selects global top w by score.

    score(S||v) = -(Rs + logP(v|S)) / (t+1) + sigma(S) / tau
    and sigma(S) is computed from current counts and does NOT depend on v (deferred calc).
    """
    model.eval()
    if not (hasattr(model, "hf_device_map") and getattr(model, "hf_device_map", None)):
        model.to(device)

    if hasattr(model.config, "num_experts"):
        num_experts = int(model.config.num_experts)
    elif hasattr(model.config, "n_experts"):
        num_experts = int(model.config.n_experts)
    elif hasattr(model.config, "num_local_experts"):
        num_experts = int(model.config.num_local_experts)
    else:
        raise ValueError("Cannot infer num_experts from config; please set it manually.")

    gate_hook = GateHook()
    gates = find_moe_gates(model, num_experts=num_experts)
    if len(gates) == 0:
        raise ValueError("No gate/router Linear found. Print model.named_modules() and adjust find_moe_gates().")

  
    layer_ids = sorted({g[0] for g in gates})
    layer_id_to_compact = {lid: i for i, lid in enumerate(layer_ids)}
    L = len(layer_ids)
    top_k = infer_top_k(model)

    handles = []
    gate_slot_to_layer_compact: List[int] = []
    for gate_slot, (lid, gate_linear, name) in enumerate(gates):
        compact = layer_id_to_compact[lid]
        gate_slot_to_layer_compact.append(compact)
        handles.append(gate_linear.register_forward_hook(gate_hook.make_hook(gate_slot)))

    try:
        bos = tokenizer.bos_token_id
        if bos is None:
            bos = tokenizer.eos_token_id
        input_device = get_input_device(model, fallback=device)
        init_ids = torch.tensor([[bos]], dtype=torch.long, device=input_device)
        eos = tokenizer.eos_token_id

        # generate each sequence independently to get num_seqs samples
        all_samples: List[torch.LongTensor] = []

        # progress helpers
        if progress and tqdm is not None:
            seq_iter = tqdm(range(num_seqs), desc="EBSS calib", unit="seq")
            log_every = None
        else:
            seq_iter = range(num_seqs)
            log_every = max(1, num_seqs // 10) if num_seqs > 0 else 1

        for seq_idx in seq_iter:
            # ---- warmup: normal sampling for first N tokens ----
            warmup_tokens_eff = max(0, int(warmup_tokens))
            input_ids = init_ids.clone()
            pkv = None
            counts_device = _infer_counts_device(model, gates=gates, fallback=input_device)
            counts = torch.zeros((1, L, num_experts), dtype=torch.float32, device=counts_device)
            cum_logprob = 0.0

            token_pbar = None
            if progress and show_token_progress and tqdm is not None:
                token_pbar = tqdm(total=seqlen - 1, desc=f"seq {seq_idx+1}/{num_seqs}", unit="tok", leave=False)

            for _t in range(warmup_tokens_eff):
                gate_hook.clear()
                if pkv is None:
                    out = model(input_ids=input_ids, use_cache=True)
                else:
                    out = model(input_ids=input_ids[:, -1:], past_key_values=pkv, use_cache=True)

                logits = out.logits[:, -1, :]  # [1, V]
                pkv = out.past_key_values

                logprobs = torch.log_softmax(logits / max(temperature, 1e-6), dim=-1)
                tok = _sample_from_topk_logprobs(logprobs, k=vocab_topk)
                # update cum_logprob with the sampled token
                cum_logprob += float(logprobs[0, tok].item())

                input_ids = torch.cat([input_ids, torch.tensor([[tok]], device=input_device)], dim=1)

                # still record routing counts during warmup (only the score ignores sigma)
                step_counts = _step_counts_from_gate_logits(
                    gate_logits=gate_hook.gate_logits,
                    gate_slot_to_layer_compact=gate_slot_to_layer_compact,
                    L=L,
                    num_experts=num_experts,
                    top_k=top_k,
                    batch_size=1,
                    device=counts_device,
                    dtype=counts.dtype,
                )
                counts += step_counts

                if stop_on_eos and eos is not None and tok == eos:
                    if token_pbar is not None:
                        token_pbar.update(1)
                    break
                if token_pbar is not None:
                    token_pbar.update(1)

            # ---- beam search main loop (BATCHED): one forward per step for all beams ----
            # warmup produced a single prefix; expand to w beams
            counts = counts.repeat(w, 1, 1)  # [w, L, E]
            sigma = compute_sigma_from_counts_batched(counts)  # [w] on counts_device
            input_ids = input_ids.repeat(w, 1)  # [w, T]
            # cache batch becomes w (prefer new Cache API). If repeating isn't supported, rebuild cache from prefix.
            try:
                pkv = _repeat_past_key_values(model, pkv, repeat=w)
            except Exception:
                pkv = None
            if pkv is None:
                # Fallback: rebuild a batched cache by running the full prefix once.
                # This is important for models using new Cache types where repeating isn't available.
                gate_hook.clear()
                out_pref = model(input_ids=input_ids, use_cache=True)
                pkv = out_pref.past_key_values

            # keep cum_logprob on the same device as logits for fast scoring
            # (small host->device copy; negligible compared to model forward)
            cum = None  # lazy init after first logits to pick correct device

            # already have input length = 1 + warmup_tokens_eff (or shorter if eos)
            for _t in range(input_ids.size(1) - 1, seqlen - 1):
                gate_hook.clear()
                if pkv is None:
                    out = model(input_ids=input_ids, use_cache=True)
                else:
                    out = model(input_ids=input_ids[:, -1:], past_key_values=pkv, use_cache=True)

                logits = out.logits[:, -1, :]  # [w, V]
                logprobs = torch.log_softmax(logits / max(temperature, 1e-6), dim=-1)  # [w, V]
                K = min(vocab_topk, logprobs.size(-1))
                topv, topi = torch.topk(logprobs, k=K, dim=-1)  # [w, K]

                if cum is None:
                    cum = torch.full((w,), float(cum_logprob), device=topv.device, dtype=torch.float32)
                else:
                    cum = cum.to(topv.device)

                sigma_score = sigma.to(topv.device)  # [w]
                nlen = input_ids.size(1) + 1
                scores = -((cum[:, None] + topv) / float(nlen)) + (sigma_score[:, None] / float(tau))  # [w, K]
                flat = scores.reshape(-1)  # [w*K]
                chosen_flat = torch.topk(flat, k=w, largest=False).indices  # [w]
                parent = (chosen_flat // K).to(dtype=torch.long)  # [w]
                choice = (chosen_flat % K).to(dtype=torch.long)   # [w]

                tok = topi[parent, choice]  # [w]
                lp = topv[parent, choice]   # [w]

                # select / duplicate beams (supports multiple children from same parent)
                input_ids = input_ids.index_select(0, parent.to(input_ids.device))
                tok_in = tok.to(input_ids.device)
                input_ids = torch.cat([input_ids, tok_in.unsqueeze(1)], dim=1)

                pkv = _select_past_key_values(model, out.past_key_values, indices=parent, batch_size=w)

                # routing stats update (vectorized scatter_add, no .tolist())
                step_counts_all = _step_counts_from_gate_logits(
                    gate_logits=gate_hook.gate_logits,
                    gate_slot_to_layer_compact=gate_slot_to_layer_compact,
                    L=L,
                    num_experts=num_experts,
                    top_k=top_k,
                    batch_size=w,
                    device=counts_device,
                    dtype=counts.dtype,
                )  # [w, L, E] on counts_device

                parent_counts = parent.to(counts_device)
                counts = counts.index_select(0, parent_counts) + step_counts_all.index_select(0, parent_counts)
                sigma = compute_sigma_from_counts_batched(counts)  # [w] on counts_device

                cum = cum.index_select(0, parent.to(cum.device)) + lp  # [w] on logits device

                # if stop_on_eos and eos is not None:
                #     # if best beam already ended, stop early (we keep chosen beams sorted by score)
                #     if int(tok[0].item()) == int(eos):
                #         if token_pbar is not None:
                #             token_pbar.update(1)
                #         break
                # if token_pbar is not None:
                #     token_pbar.update(1)
                
                if stop_on_eos and eos is not None:
                    # Check if any beam hit EOS by examining the last token in each beam
                    # input_ids[:, -1] gives the last token of each beam [w]
                    # This avoids device sync issues by using tensor operations
                    last_tokens = input_ids[:, -1]  # [w]
                    if torch.any(last_tokens == eos):
                        if token_pbar is not None:
                            token_pbar.update(1)
                        break
                if token_pbar is not None:
                    token_pbar.update(1)

            # take the best beam as one calibration sample
            if cum is None:
                # degenerate: no beam steps happened; fall back to the prefix
                all_samples.append(input_ids[0].detach().cpu())
            else:
                final = -(cum / float(input_ids.size(1))) + (sigma.to(cum.device) / float(tau))  # [w]
                best = int(torch.argmin(final).item())
                all_samples.append(input_ids[best].detach().cpu())

            if token_pbar is not None:
                token_pbar.close()

            if log_every is not None and ((seq_idx + 1) % log_every == 0 or (seq_idx + 1) == num_seqs):
                print(f"[EBSS calib] generated {seq_idx + 1}/{num_seqs} sequences")

        return all_samples

    finally:
        for h in handles:
            h.remove()

def save_calib(samples: List[torch.LongTensor], path: str):
    dir_path = os.path.dirname(path)
    if dir_path:  # If the path includes a directory component
        os.makedirs(dir_path, exist_ok=True)
    torch.save({"input_ids": samples}, path)

def save_calib_jsonl(samples, tokenizer, path: str):
        # Ensure the output directory exists
    dir_path = os.path.dirname(path)
    if dir_path:  # If the path includes a directory component
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ids in samples:
            text = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            obj = {"text": text}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
def main():
    ap = argparse.ArgumentParser(description="EBSS-style calibration data generation with MoE expert-balance penalty.")
    ap.add_argument("--model", type=str, required=True, help="HF repo id or local path")
    ap.add_argument("--out", type=str, required=True, help="output .pt path (torch.save dict with input_ids)")
    ap.add_argument("--out_jsonl", type=str, required=True, help="output .jsonl path")
    ap.add_argument("--num_seqs", type=int, default=150)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--w", type=int, default=2, help="beam width")
    ap.add_argument("--tau", type=float, default=1.2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--vocab_topk", type=int, default=50)
    ap.add_argument("--warmup_tokens", type=int, default=3, help="first N tokens use normal sampling (no sigma influence)")
    ap.add_argument("--stop_on_eos", action="store_true", default=False)
    ap.add_argument("--no_progress", action="store_true", default=False, help="disable progress output")
    ap.add_argument("--show_token_progress", action="store_true", default=True, help="show per-token progress (slower)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", type=str, default="auto", help='e.g. "auto" for large models; set to "none" to disable')
    args = ap.parse_args()

    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers is required to run this script: pip install transformers")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(dtype=torch_dtype, trust_remote_code=args.trust_remote_code)
    device_map = None if (args.device_map is None or str(args.device_map).lower() == "none") else args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=device_map, **kwargs)
   
    if getattr(model, "hf_device_map", None):
        dm = model.hf_device_map
        from collections import Counter
        counts = Counter(str(v) for v in dm.values())
        print(f"[load] hf_device_map device counts: {dict(counts)}")
        print(f"[load] hf_device_map example entries: {list(dm.items())[:5]}")
        if any("cpu" in k for k in counts.keys()) or any("disk" in k for k in counts.keys()):
            print("[load] WARNING: some modules are on CPU/disk (offloaded). Generation may be very slow.")
    else:
        try:
            print(f"[load] model device = {next(model.parameters()).device}")
        except Exception:
            pass

    samples = ebss_generate_calib(
        model=model,
        tokenizer=tokenizer,
        num_seqs=args.num_seqs,
        seqlen=args.seqlen,
        w=args.w,
        tau=args.tau,
        temperature=args.temperature,
        vocab_topk=args.vocab_topk,
        device=args.device,
        warmup_tokens=args.warmup_tokens,
        stop_on_eos=args.stop_on_eos,
        progress=not args.no_progress,
        show_token_progress=args.show_token_progress,
    )
    save_calib(samples, args.out)
    save_calib_jsonl(samples,tokenizer,args.out_jsonl)

if __name__ == "__main__":
    main()
