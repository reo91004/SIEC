# [EXP] /home/user/jowithu/Semantic/IEC/mainddpm/ddim_cifar_siec.py 의 실험용 복사본.
# 원본은 수정하지 않음. 아래 수정 부분은 # [EXP-CN] 태그로 표시.
# 이 파일 디렉토리를 sys.path 맨 앞에 넣어 `deepcache` / `deepcache_denoising`
# import 가 flat 복사본으로 라우팅되게 한다. 나머지는 ./mainddpm 경로로 해소.
import sys
import pathlib
# [EXP] shim: 이 파일 디렉토리를 가장 먼저 배치해 `from deepcache import Diffusion`
# 가 같은 폴더의 deepcache.py 를 읽고, ddpm/runners/deepcache 는 가리지 않는다.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.append("./mainldm")
sys.path.append("./mainddpm")
sys.path.append('./src/taming-transformers')
sys.path.append('.')
print(sys.path)
import argparse
import traceback
import shutil
import logging
import yaml
import random
import os, logging, gc
import torch
import numpy as np
from tqdm import tqdm

from ddpm.utils.tools import set_random_seed
from accelerate import Accelerator, DistributedDataParallelKwargs
from quant.utils import AttentionMap, seed_everything, Fisher 
from quant.quant_model import QModel
from quant.set_quantize_params import set_act_quantize_params, set_weight_quantize_params
from quant.recon_Qmodel import recon_Qmodel, skip_Model

import matplotlib.pyplot as plt
torch.set_printoptions(sci_mode=False)
logger = logging.getLogger(__name__)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def block_train_w(q_unet, args, kwargs, cali_data, t, cali_t, cache):
    recon_qnn = recon_Qmodel(args, q_unet, kwargs)
    q_unet.block_count = 0
    kwargs['cali_data'] = (cali_data, t, cache)
    kwargs['cali_t'] = cali_t
    kwargs['branch'] = args.branch
    recon_qnn.kwargs = kwargs
    recon_qnn.down_name = None
    del (cali_data, t, cache)
    gc.collect()
    q_unet.set_steps_state(is_mix_steps=True)
    q_unet = recon_qnn.recon()
    q_unet.set_steps_state(is_mix_steps=False)
    torch.cuda.empty_cache()


def normalize_correction_mode(args) -> str:
    if args.correction_mode != "auto":
        return args.correction_mode
    return "siec" if args.use_siec else "iec"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--config", type=str, default="./mainddpm/configs/cifar10.yml")
    parser.add_argument("--seed", type=int, default=1234+9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp", type=str, default="deepcache")
    parser.add_argument("--image_folder", type=str, default="./error_dec/cifar/image_siec",
                        help="folder for S-IEC samples (distinct from IEC's image/)")
    parser.add_argument("--fid", action="store_true", default=True)
    parser.add_argument("--interpolation", action="store_true", default=False)
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--ni", action="store_true", default=True)
    parser.add_argument("--use_pretrained", action="store_true", default=True)
    parser.add_argument("--sample_type", type=str, default="generalized")
    parser.add_argument("--skip_type", type=str, default="quad")
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--select_step", type=int, default=None)
    parser.add_argument("--select_depth", type=int, default=None)
    parser.add_argument("--cache", action="store_true", default=True)
    # [EXP-C4] DeepCache 비활성 옵션 (실험 5 Setting 1 fp16 용).
    parser.add_argument("--no-cache", dest="cache", action="store_false")
    parser.add_argument("--replicate_interval", type=int, default=10)
    parser.add_argument("--non_uniform", action="store_true", default=False)
    parser.add_argument("--pow", type=float, default=None)
    parser.add_argument("--center", type=int, default=None)
    parser.add_argument("--branch", type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--sample_batch', type=int, default=500)

    parser.add_argument("--sm_abit", type=int, default=8)
    parser.add_argument("--quant_act", action="store_true", default=True)
    parser.add_argument("--weight_bit", type=int, default=8)
    parser.add_argument("--act_bit", type=int, default=8)
    parser.add_argument("--quant_mode", type=str, default="qdiff", choices=["qdiff"])
    parser.add_argument("--lr_w", type=float, default=1e-4)
    parser.add_argument("--lr_a", type=float, default=1e-4)
    parser.add_argument("--lr_z", type=float, default=1e-4)
    parser.add_argument("--split", action="store_true", default=True)
    parser.add_argument("--ptq", action="store_true", default=True)
    # [EXP-C4] PTQ 양자화 비활성 옵션 (실험 5 Setting 1 fp16 용).
    parser.add_argument("--no-ptq", dest="ptq", action="store_false")
    parser.add_argument("--dps_steps", action="store_true", default=False)
    parser.add_argument("--recon", action="store_true", default=False)

    # ========== S-IEC specific arguments ==========
    parser.add_argument("--use_siec", action="store_true", default=True,
                        help="Enable S-IEC (syndrome-guided correction)")
    # [EXP-C2] IEC-only fresh run 을 위한 S-IEC 비활성 옵션 (실험 5).
    parser.add_argument("--no-use-siec", dest="use_siec", action="store_false")
    # `siec_oracle` added by [EXP-FRAMING-E] for oracle decoder ablation.
    parser.add_argument("--correction-mode", type=str, default="auto",
                        choices=["auto", "none", "tac", "iec", "siec", "siec_oracle"],
                        help="Explicit correction mode. 'auto' keeps legacy use_siec semantics.")
    parser.add_argument("--c_siec", type=float, default=1.0,
                        help="Correction gain multiplier (c in lambda = c*sigma^2/(alpha^2+sigma^2))")
    parser.add_argument("--tau_path", type=str,
                        default="./calibration/tau_schedule_p80.pt",
                        help="Pre-calibrated tau schedule (.pt)")
    parser.add_argument("--tau_percentile", type=float, default=80.0,
                        help="Percentile used for calibration (metadata only)")
    parser.add_argument("--siec_always_correct", action="store_true", default=False,
                        help="Ablation: always correct, no threshold triggering")
    parser.add_argument("--siec_collect_scores", action="store_true", default=False,
                        help="Pilot mode: collect syndrome scores without correcting")
    parser.add_argument("--siec_scores_out", type=str,
                        default="./calibration/pilot_scores.pt",
                        help="Where to save collected scores in pilot mode")
    parser.add_argument("--siec_max_rounds", type=int, default=1,
                    help="Number of inner correction rounds (toy SIEC.max_rounds)")
    # [EXP-C1] 실험 4 트리거 모드 베이스라인 (random/uniform vs syndrome).
    parser.add_argument("--trigger_mode", type=str, default="syndrome",
                        choices=["syndrome", "random", "uniform"])
    parser.add_argument("--trigger_prob", type=float, default=0.2,
                        help="random-mode: per-step trigger probability")
    parser.add_argument("--trigger_period", type=int, default=5,
                        help="uniform-mode: cur_i %% period == 0 triggers")
    # [EXP-C3] NFE/trigger 통계 직접 측정용 adaptive_generalized_steps_trace API 노출.
    parser.add_argument("--siec_return_trace", action="store_true", default=False)
    parser.add_argument("--siec_trace_out", type=str,
                        default="./calibration/siec_trace.pt")
    parser.add_argument("--siec_trace_mode", type=str, default="auto",
                        choices=["auto", "none", "tac", "iec", "siec"])
    parser.add_argument("--trace_include_x0", action="store_true", default=False,
                        help="Include per-step x0 trajectory in the saved trace (large files).")
    parser.add_argument("--disable-cache-reuse", action="store_true", default=False,
                        help="Keep DeepCache calibration/intervals but force slow-path forwards at every step.")
    # [EXP-FRAMING-A] xt trajectory 와 et per step 도 trace 에 포함 (실험 A 의 deploy vs ref 거리 측정용).
    parser.add_argument("--trace_include_xs", action="store_true", default=False,
                        help="[EXP-FRAMING-A] Include per-step xt trajectory and et in the saved trace.")
    # [EXP-FRAMING-D] toy 의 net 1-NFE 의도를 CIFAR 로 옮긴 lookahead 재활용 (cache reuse + memoization).
    parser.add_argument("--reuse_lookahead", action="store_true", default=False,
                        help="[EXP-FRAMING-D] Reuse lookahead forward in next step's first forward (toy intent).")
    # [EXP-FRAMING-E] Oracle decoder: per-step xt 를 fp16 reference 로 직접 pull.
    parser.add_argument("--oracle_xt_ref", type=str, default=None,
                        help="[EXP-FRAMING-E] Path to .pt with reference xs_trajectory (list of CPU tensors).")
    parser.add_argument("--oracle_pull_strength", type=float, default=1.0,
                        help="[EXP-FRAMING-E] Pull strength (0=no pull, 1=full ref). Used by correction_mode=siec_oracle.")
    # NOTE: The earlier `--correction-mode` (line ~122) already covers `siec_oracle`
    # — do not register a second `--correction_mode` flag (argparse would silently let
    # the last-in-argv one win, which is a footgun).
    # ===============================================

    args = parser.parse_args()

    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)

    if args.dps_steps:
        args.mode = "dps_opt"
    else:
        args.mode = "uni"

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    new_config.select_step = args.select_step
    new_config.select_depth = args.select_depth
    torch.backends.cudnn.benchmark = True

    args, config = args, new_config
    accelerator = Accelerator()
    args.accelerator = accelerator
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("./run.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logging.info("start!")
    seed_everything(args.seed)
    args.correction_mode = normalize_correction_mode(args)
    if args.siec_collect_scores and args.correction_mode != "siec":
        raise ValueError("--siec_collect_scores requires correction-mode=siec")

    # Experimental copy keeps DeepCache calibration/intervals available even when
    # cache reuse is disabled. This is how exp5 pure-quant / fp16 settings reuse
    # the same schedule/check positions without inheriting cache-reuse error.
    if args.disable_cache_reuse and not args.cache:
        raise ValueError("use either --disable-cache-reuse or --no-cache, not both")
    if args.ptq and not args.cache:
        raise ValueError("PTQ with --no-cache is unsupported in this experimental copy; use --disable-cache-reuse instead")
    if args.correction_mode != "none" and not args.cache:
        raise ValueError("correction-mode with --no-cache is unsupported; keep cache enabled and use --disable-cache-reuse if needed")

    load_calibration = args.cache or args.disable_cache_reuse or args.ptq or args.correction_mode in {"iec", "siec", "tac"}
    if load_calibration:
        logger.info("load calibration...")
        interval_seq, all_cali_data, all_t, all_cali_t, all_cache = torch.load(
            "./calibration/cifar{}_cache{}_{}.pth".format(
                args.timesteps, args.replicate_interval, args.mode
            )
        )
        args.interval_seq = interval_seq
        logger.info(f"The interval_seq: {args.interval_seq}")
    else:
        logger.info("[EXP] calibration not required for this run.")
        interval_seq, all_cali_data, all_t, all_cali_t, all_cache = None, [], [], [], []
        args.interval_seq = None

    # ============== S-IEC tau schedule ==============
    if args.correction_mode == "siec":
        if args.siec_collect_scores:
            logger.info("[S-IEC] Pilot mode: will collect syndrome scores (no correction).")
            args.tau_schedule = None
        elif args.siec_always_correct:
            logger.info("[S-IEC] Always-correct mode (ablation): no threshold triggering.")
            args.tau_schedule = None
        elif os.path.exists(args.tau_path):
            logger.info(f"[S-IEC] Loading tau schedule from {args.tau_path}")
            tau = torch.load(args.tau_path)
            if isinstance(tau, torch.Tensor):
                tau = tau.cpu().numpy()
            args.tau_schedule = tau
            logger.info(f"[S-IEC]   tau shape: {tau.shape}, mean: {tau.mean():.6f}, "
                        f"min: {tau.min():.6f}, max: {tau.max():.6f}")
        else:
            logger.warning(f"[S-IEC] tau_path not found: {args.tau_path}")
            logger.warning(f"[S-IEC] Falling back to always-correct mode.")
            args.tau_schedule = None
            args.siec_always_correct = True
    else:
        args.tau_schedule = None
    # =================================================

    # [EXP] 같은 폴더의 deepcache.py (C1/C3/C4 패치 포함) 로 라우팅.
    from deepcache import Diffusion
    runner = Diffusion(args, config, interval_seq=args.interval_seq)
    model = runner.creat_model()

    # [EXP-C4] a/b DEC 리스트는 cache 활성 run 에서만 존재.
    if args.cache:
        (a_list, b_list) = torch.load(
            f"./error_dec/cifar/pre_cacheerr_abCov_interval{args.replicate_interval}_list_timesteps{args.timesteps}.pth")
        model.a_list = torch.stack(a_list)
        model.b_list = torch.stack(b_list)
    model.timesteps = args.timesteps

    if args.ptq:
        wq_params = {'n_bits': args.weight_bit, 'symmetric': False, 'channel_wise': True, 'scale_method': 'mse'}
        aq_params = {'n_bits': args.act_bit, 'symmetric': False, 'channel_wise': False,
                     'scale_method': 'mse', 'leaf_param': args.quant_act,
                     "prob": 1.0, "num_timesteps": args.timesteps}
        q_unet = QModel(model, args, wq_params=wq_params, aq_params=aq_params)
        q_unet.cuda()
        q_unet.eval()
        q_unet._exp_backend = getattr(model, "_exp_backend", "deepcache")

        print("Setting the first and the last layer to 8-bit")
        q_unet.set_first_last_layer_to_8bit()
        q_unet.set_quant_state(False, False)

        if args.split == True:
            q_unet.model.config.split_shortcut = True

        cali_data = torch.cat(all_cali_data)
        t = torch.cat(all_t)
        idx = torch.randperm(len(cali_data))[:32]
        cali_data = cali_data[idx]
        t = t[idx]

        set_weight_quantize_params(q_unet, cali_data=(cali_data, t))
        set_act_quantize_params(args.interval_seq, q_unet, all_cali_data, all_t, all_cache)

        if not args.recon:
            pre_err_list = torch.load(
                f"./error_dec/cifar/pre_quanterr_abCov_weight{args.weight_bit}_interval{args.replicate_interval}_list_timesteps{args.timesteps}.pth")
            q_unet.model.up[1].block[2].nin_shortcut.pre_err = pre_err_list

        q_unet.set_quant_state(True, True)

        if args.recon:
            skip_model = skip_Model(q_unet)
            q_unet = skip_model.set_skip()
            kwargs = dict(iters=3000, act_quant=True, weight_quant=True, asym=True,
                          opt_mode='mse', lr_z=args.lr_z, lr_a=args.lr_a, lr_w=args.lr_w,
                          p=2.0, weight=0.01, b_range=(20, 2), warmup=0.2,
                          batch_size=32, batch_size1=64, input_prob=1.0,
                          recon_w=True, recon_a=True, keep_gpu=False,
                          interval_seq=args.interval_seq, weight_bits=args.weight_bit)
            all_cali_data = torch.cat(all_cali_data)
            all_t = torch.cat(all_t)
            all_cali_t = torch.cat(all_cali_t)
            all_cache = torch.cat(all_cache)
            idx = torch.randperm(len(all_cali_data))[:1024]
            cali_data = all_cali_data[idx].detach()
            t = all_t[idx].detach()
            cali_t = all_cali_t[idx].detach()
            cache = all_cache[idx].detach()
            del (all_cali_data, all_t, all_cali_t, all_cache)
            gc.collect()
            q_unet.set_quant_state(weight_quant=True, act_quant=args.quant_act)
            block_train_w(q_unet, args, kwargs, cali_data, t, cali_t, cache)
            q_unet.set_quant_state(weight_quant=True, act_quant=args.quant_act)

        seed_everything(args.seed)
        model.time = 0
        runner.sample_fid(q_unet, total_n_samples=args.num_samples)
    else:
        seed_everything(args.seed)
        runner.sample_fid(model, total_n_samples=args.num_samples)

    # ========== Save pilot scores if in pilot mode ==========
    if args.correction_mode == "siec" and args.siec_collect_scores:
        if hasattr(args, '_pilot_scores') and len(args._pilot_scores) > 0:
            logger.info(f"[S-IEC] Pilot collected {len(args._pilot_scores)} batches of scores")
            # Preserve both per-sample score distributions (for tau calibration)
            # and per-batch mean scores (for trigger-rate matching against the
            # runtime, which currently triggers on batch-mean score > tau).
            T = len(args._pilot_scores[0])
            scores_by_t = [[] for _ in range(T)]
            batch_score_means_by_t = [[] for _ in range(T)]
            for batch_scores in args._pilot_scores:
                for t_idx, step_scores in enumerate(batch_scores):
                    if step_scores is None:
                        continue
                    if isinstance(step_scores, (list, tuple)):
                        values = [float(s) for s in step_scores]
                    else:
                        values = [float(step_scores)]
                    if not values:
                        continue
                    scores_by_t[t_idx].extend(values)
                    batch_score_means_by_t[t_idx].append(sum(values) / len(values))

            os.makedirs(os.path.dirname(args.siec_scores_out), exist_ok=True)
            torch.save(
                {
                    "format_version": 2,
                    "scores_by_t": scores_by_t,
                    "batch_score_means_by_t": batch_score_means_by_t,
                    "num_batches": len(args._pilot_scores),
                    "num_timesteps": T,
                },
                args.siec_scores_out,
            )
            logger.info(f"[S-IEC] Saved pilot scores to {args.siec_scores_out}")
        else:
            logger.warning("[S-IEC] Pilot mode but no scores collected!")

    # [EXP-C3] --siec_return_trace 일 때 step 별 트리거 수 / NFE trace 를 저장.
    if args.siec_return_trace:
        traces = getattr(args, "_exp_traces", [])
        if traces:
            os.makedirs(os.path.dirname(args.siec_trace_out), exist_ok=True)
            torch.save(traces, args.siec_trace_out)
            logger.info(f"[EXP-C3] trace dict {len(traces)} 개 저장 → {args.siec_trace_out}")
        else:
            logger.warning("[EXP-C3] trace 모드인데 수집된 trace 없음!")
    # ===========================================================

    logging.info("sample siec finish!")
