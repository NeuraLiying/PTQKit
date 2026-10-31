import os
import sys
import time
import torch
from torch import nn
import numpy as np
from functools import partial
import argparse
import importlib
import timm
import copy
from datetime import datetime
import logging


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FIMAQ_ROOT = os.path.dirname(_THIS_DIR)
if _FIMAQ_ROOT not in sys.path:
    sys.path.insert(0, _FIMAQ_ROOT)

import utils.datasets as mydatasets
from utils.calibrator import QuantCalibrator
from utils.block_recon import BlockReconstructor
from utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net
from utils.test_utils import validate


root_path = None

import builtins

original_print = builtins.print

def custom_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    original_print(*args, **kwargs)


builtins.print = custom_print


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model",
        default="fastvit_ma36",
        type=str,
        help="timm model name, e.g. fastvit_t8 / fastvit_t12 / fastvit_s12 ...",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_FIMAQ_ROOT, "configs/4bit/best.py"),
        help="File path to import Config class from",
    )
    parser.add_argument("--dataset", default="/path/to/ImageNet_ILSVRC2012/", help="path to dataset")
    parser.add_argument("--calib-size", default=argparse.SUPPRESS, type=int, help="size of calibration set")
    parser.add_argument("--optim-size", default=1024, type=int, help="size of optimization set")
    parser.add_argument("--calib-batch-size", default=argparse.SUPPRESS, type=int, help="batchsize of calibration set")
    parser.add_argument("--optim-batch-size", default=argparse.SUPPRESS, type=int, help="batchsize of optimization set")
    parser.add_argument("--val-batch-size", default=200, type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=8, type=int, help="number of data loading workers (default: 8)")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument(
        "--skip-fp-eval",
        action="store_true",
        help="Skip FP baseline ImageNet validation (default: evaluate FP before quantizing)",
    )
    parser.add_argument(
        "--quant-act",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to quantize activations",
    )
    parser.add_argument("--recon-iters", type=int, default=20000, help="Block-reconstruction iterations (start small for FastViT)")
    parser.add_argument("--recon-batch-size", type=int, default=32, help="Block-reconstruction batch size")
    parser.add_argument("--recon-lr", type=float, default=4e-5, help="Block-reconstruction learning rate")
    parser.add_argument("--recon-weight", type=float, default=0.01, help="Rounding regularization weight")
    parser.add_argument("--recon-block-limit", type=int, default=0, help="Reconstruct only the first N blocks (0 = all)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional local checkpoint to initialize the FP model (passed to timm.create_model(checkpoint_path=...))",
    )

    calibrate_mode_group = parser.add_mutually_exclusive_group()
    calibrate_mode_group.add_argument("--calibrate", action="store_true", help="Calibrate the model")
    calibrate_mode_group.add_argument(
        "--load-calibrate-checkpoint", type=str, default=None, help="Path to the calibrated checkpoint."
    )
    parser.add_argument("--test-calibrate-checkpoint", action="store_true", help="validate the calibrated checkpoint.")

    optimize_mode_group = parser.add_mutually_exclusive_group()
    optimize_mode_group.add_argument("--optimize", action="store_true", help="Optimize the model")
    optimize_mode_group.add_argument(
        "--load-optimize-checkpoint", type=str, default=None, help="Path to the optimized checkpoint."
    )
    parser.add_argument("--test-optimize-checkpoint", action="store_true", help="validate the optimized checkpoint.")

    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--seed", default=3407, type=int, help="seed")
    parser.add_argument("--w_bit", type=int, default=argparse.SUPPRESS, help="bit-precision of weights")
    parser.add_argument("--a_bit", type=int, default=argparse.SUPPRESS, help="bit-precision of activation")
    parser.add_argument(
        "--calib-metric",
        type=str,
        default=argparse.SUPPRESS,
        choices=["mse", "mae"],
        help="calibration metric",
    )
    parser.add_argument(
        "--optim-metric",
        type=str,
        default=argparse.SUPPRESS,
        choices=["fisher_brecq", "fisher_lr", "fisher_diag", "fisher_dplr", "mse", "mae"],
        help="optimization metric",
    )
    parser.add_argument(
        "--optim-mode",
        type=str,
        default=argparse.SUPPRESS,
        choices=["qinp", "rinp", "qdrop"],
        help="`qinp`:use quanted input; `rinp`: use raw input; `qdrop` use qdrop input;",
    )
    parser.add_argument(
        "--drop-prob",
        type=float,
        default=argparse.SUPPRESS,
        help="dropping rate in qdrop. set `drop-prob = 1.0` if do not use qdrop.",
    )
    parser.add_argument("--k", type=int, default=argparse.SUPPRESS, help="The rank of Fisher")
    parser.add_argument("--p1", type=float, default=argparse.SUPPRESS, help="The proportion of low rank")
    parser.add_argument("--p2", type=float, default=argparse.SUPPRESS, help="The proportion of diag")
    parser.add_argument(
        "--dis-mode",
        type=str,
        default=argparse.SUPPRESS,
        choices=["q", "qf"],
        help="the mode of getting gradient. `q`: use quantization; `qf` Take the first k times (default:Uniformly obtain k times);",
    )
    return parser


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_cur_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_model(model, args, cfg, mode="calibrate"):
    assert mode in ["calibrate", "optimize"]
    if mode == "calibrate":
        auto_name = "{}_w{}_a{}_calibsize_{}_{}.pth".format(args.model, cfg.w_bit, cfg.a_bit, cfg.calib_size, cfg.calib_metric)
    else:
        auto_name = "{}_w{}_a{}_optimsize_{}_{}{}{}_{}.pth".format(
            args.model,
            cfg.w_bit,
            cfg.a_bit,
            cfg.optim_size,
            cfg.optim_metric,
            "" if cfg.optim_metric in ["mse", "mae"] else "_dis_mode_" + cfg.dis_mode,
            "" if cfg.optim_metric not in ["fisher_lr", "fisher_dplr"] else "_rank_" + str(cfg.k),
            cfg.optim_mode,
        )
    save_path = os.path.join(root_path, auto_name)
    logging.info(f"Saving checkpoint to {save_path}")
    torch.save(model.state_dict(), save_path)


def load_model(model, args, device, mode="calibrate"):
    assert mode in ["calibrate", "optimize"]
    ckpt_path = args.load_calibrate_checkpoint if mode == "calibrate" else args.load_optimize_checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    for name, module in model.named_modules():
        if hasattr(module, "mode"):
            module.calibrated = True
            module.mode = "quant_forward"
        if isinstance(module, nn.Linear) and "reduction" in name:
            module.bias = nn.Parameter(torch.zeros(module.out_features))
        # Handle checkpoints that include qkv.bias when the current model does not (FIMA-Q version skew)
        if isinstance(module, nn.Linear) and "qkv" in name:
            bias_key = name + ".bias"
            if bias_key in ckpt and (module.bias is None or module.bias.shape != ckpt[bias_key].shape):
                module.bias = nn.Parameter(ckpt[bias_key].clone())
                logging.info(f"Restored {bias_key} from checkpoint")
        quantizer_attrs = ["a_quantizer", "w_quantizer", "A_quantizer", "B_quantizer"]
        for attr in quantizer_attrs:
            if hasattr(module, attr):
                getattr(module, attr).inited = True
                ckpt_name = name + "." + attr + ".scale"
                if ckpt_name in ckpt:
                    getattr(module, attr).scale.data = ckpt[ckpt_name].clone()

    result = model.load_state_dict(ckpt, strict=False)
    logging.info(str(result))
    model.to(device)
    model.eval()
    return model


def _build_model(args):
    if args.checkpoint:
        return timm.create_model(args.model, pretrained=False, checkpoint_path=args.checkpoint)
    return timm.create_model(args.model, pretrained=True)



def main(args):
    logging.info("{} - start the process.".format(get_cur_time()))
    logging.info(str(args))

    dir_path = os.path.dirname(os.path.abspath(args.config))
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    module_name = os.path.splitext(os.path.basename(args.config))[0]
    imported_module = importlib.import_module(module_name)
    Config = getattr(imported_module, "Config")
    logging.info("Successfully imported Config class!")

    cfg = Config()
    cfg.calib_size = args.calib_size if hasattr(args, "calib_size") else cfg.calib_size
    cfg.optim_size = args.optim_size if hasattr(args, "optim_size") else cfg.optim_size
    cfg.calib_batch_size = args.calib_batch_size if hasattr(args, "calib_batch_size") else cfg.calib_batch_size
    cfg.optim_batch_size = args.optim_batch_size if hasattr(args, "optim_batch_size") else cfg.optim_batch_size
    cfg.calib_metric = args.calib_metric if hasattr(args, "calib_metric") else cfg.calib_metric
    cfg.optim_metric = args.optim_metric if hasattr(args, "optim_metric") else cfg.optim_metric
    cfg.optim_mode = args.optim_mode if hasattr(args, "optim_mode") else cfg.optim_mode
    cfg.drop_prob = args.drop_prob if hasattr(args, "drop_prob") else cfg.drop_prob
    cfg.w_bit = args.w_bit if hasattr(args, "w_bit") else cfg.w_bit
    cfg.a_bit = args.a_bit if hasattr(args, "a_bit") else cfg.a_bit
    cfg.k = args.k if hasattr(args, "k") else cfg.k
    cfg.p1 = args.p1 if hasattr(args, "p1") else cfg.p1
    cfg.p2 = args.p2 if hasattr(args, "p2") else cfg.p2
    cfg.dis_mode = args.dis_mode if hasattr(args, "dis_mode") else cfg.dis_mode

    # Default: leave activations in FP to avoid large FastViT accuracy drops at low activation bits
    if not args.quant_act:
        cfg.a_bit = 32
        cfg.qhead_a_bit = 32
        cfg.qconv_a_bit = 32
    for name, value in vars(cfg).items():
        logging.info(f"{name}: {value}")

    if args.device.startswith("cuda:"):
        gpu_id = args.device.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        args.device = "cuda:0"
    device = torch.device(args.device)

    seed_all(args.seed)

    logging.info("Building model ...")
    model = _build_model(args)
    model.to(device)
    model.eval()

    data_path = args.dataset
    g = mydatasets.ViTImageNetLoaderGenerator(data_path, args.val_batch_size, args.num_workers, kwargs={"model": model})

    logging.info("Building validation dataloader ...")
    val_loader = g.val_loader()
    criterion = nn.CrossEntropyLoss().to(device)

    # FP baseline
    if not args.skip_fp_eval:
        logging.info("Evaluating FP baseline ...")
        _ = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)

    reparam = args.load_calibrate_checkpoint is None and args.load_optimize_checkpoint is None
    logging.info("Wraping quantiztion modules (reparam: {}) ...".format(reparam))
    model = wrap_modules_in_net(model, cfg, reparam=reparam)
    model.to(device)
    model.eval()

    if not args.load_optimize_checkpoint:
        if args.load_calibrate_checkpoint:
            logging.info(f"Restoring checkpoint from '{args.load_calibrate_checkpoint}'")
            model = load_model(model, args, device, mode="calibrate")
            if args.test_calibrate_checkpoint:
                _ = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
        else:
            logging.info("{} - start {} guided calibration".format(get_cur_time(), cfg.calib_metric))
            calib_loader = g.calib_loader(num=cfg.calib_size, batch_size=cfg.calib_batch_size, seed=args.seed)
            quant_calibrator = QuantCalibrator(model, calib_loader)
            quant_calibrator.batching_quant_calib()
            model = wrap_reparamed_modules_in_net(model)
            model.to(device)
            logging.info("{} - {} guided calibration finished.".format(get_cur_time(), cfg.calib_metric))
            save_model(model, args, cfg, mode="calibrate")

            logging.info("Validating after calibration ...")
            _ = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)

    if args.optimize:
        logging.info("Building calibrator ...")
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        logging.info("{} - start {} guided block reconstruction".format(get_cur_time(), cfg.optim_metric))
        block_reconstructor = BlockReconstructor(
            model,
            cfg.optim_batch_size,
            calib_loader,
            metric=cfg.optim_metric,
            temp=cfg.temp,
            k=cfg.k,
            dis_mode=cfg.dis_mode,
            p1=cfg.p1,
            p2=cfg.p2,
        )
        block_reconstructor.reconstruct_model(
            quant_act=bool(args.quant_act),
            mode=cfg.optim_mode,
            drop_prob=cfg.drop_prob,
            keep_gpu=cfg.keep_gpu,
            iters=args.recon_iters,
            batch_size=args.recon_batch_size,
            lr=args.recon_lr,
            weight=args.recon_weight,
            block_limit=args.recon_block_limit,
        )
        logging.info("{} - {} guided block reconstruction finished.".format(get_cur_time(), cfg.optim_metric))
        save_model(model, args, cfg, mode="optimize")

    if args.load_optimize_checkpoint:
        logging.info("Building calibrator ...")
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        model = load_model(model, args, device, mode="optimize")

    if args.optimize or args.test_optimize_checkpoint:
        logging.info("Validating on calibration set after block reconstruction ...")
        _ = validate(calib_loader, model, criterion, print_freq=args.print_freq, device=device)
        logging.info("Validating on test set after block reconstruction ...")
        _ = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)

    logging.info("{} - finished the process.".format(get_cur_time()))


def setup_root_path_and_logging(args):
    """Choose the output directory from activation bit-width; name format: model-timestamp."""
    global root_path

    # Load config to read a_bit
    dir_path = os.path.dirname(os.path.abspath(args.config))
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    module_name = os.path.splitext(os.path.basename(args.config))[0]
    imported_module = importlib.import_module(module_name)
    Config = getattr(imported_module, "Config")
    cfg = Config()
    cfg.a_bit = args.a_bit if hasattr(args, "a_bit") else cfg.a_bit
   
    if not args.quant_act:
        cfg.a_bit = 32

    # Choose subdirectory from activation bit-width
    base_dir = os.path.abspath(os.path.join(_FIMAQ_ROOT, "checkpoints"))
    a_bit = getattr(cfg, "a_bit", None)
    if a_bit is None or a_bit >= 32:
        subdir = "weight-only"
    elif a_bit == 8:
        subdir = "W4A8"
    elif a_bit == 4:
        subdir = "W4A4"
    else:
        subdir = "W4A{}".format(a_bit)

    while True:
        try:
            timestamp = datetime.now()
            formatted_timestamp = timestamp.strftime("%Y%m%d_%H%M")
            folder_name = "{}-{}".format(args.model, formatted_timestamp)
            root_path = os.path.join(base_dir, subdir, folder_name)
            os.makedirs(root_path, exist_ok=False)
            break
        except FileExistsError:
            time.sleep(10)


    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(root_path, "output.log")),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info("[output] save directory: {}".format(root_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    setup_root_path_and_logging(args)
    main(args)

