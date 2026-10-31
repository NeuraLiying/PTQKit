    parser.add_argument("--visual_device", default=None, type=str)

    args = parser.parse_args()
    return args


# ═══════════════════════════════════════════════════════════════════════
#  ViT pseudo-quantization loading (for eval)
# ═══════════════════════════════════════════════════════════════════════
def load_quantized_vision(model, args):
    """Load quantized vision state_dict for evaluation."""
    if not args.load_vision:
        print("[ViT] No --load_vision specified, using unquantized vision.")
        return

    print(f"[ViT] Loading quantized vision from {args.load_vision}")
    # Qwen3-VL MoE nests visual under model.model (same as main_quant.py)
    if hasattr(model, "visual"):
        vision = model.visual
    elif hasattr(model, "model") and hasattr(model.model, "visual"):
        vision = model.model.visual
    elif hasattr(model, "model") and hasattr(model.model, "vision_model"):
        vision = model.model.vision_model
    else:
        raise AttributeError("Cannot find visual tower on model for --load_vision")
    state = torch.load(args.load_vision, map_location="cpu")

    if args.vit_method == "gptaq":
        add_actquant(vision)
        model_keys = set(vision.state_dict().keys())
        remapped = {}
        for k, v in state.items():
            if k in model_keys:
                remapped[k] = v
            elif ".module." not in k and (k.endswith(".weight") or k.endswith(".bias")):
                mapped = k.replace(".weight", ".module.weight") if k.endswith(".weight") else k.replace(".bias", ".module.bias")
                remapped[mapped] = v if mapped in model_keys else state[k]
            else:
                remapped[k] = v
        vision.load_state_dict(remapped, strict=False)

