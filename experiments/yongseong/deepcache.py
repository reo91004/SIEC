# [EXP] /home/user/jowithu/Semantic/IEC/mainddpm/ddpm/runners/deepcache.py 의 실험용 복사본.
# 원본은 수정하지 않음. 아래 수정 부분은 # [EXP-CN] 태그로 표시.
# flat import 구조. 엔트리인 ddim_cifar_siec.py 의 sys.path shim 이 deepcache /
# deepcache_denoising 을 이 파일로 라우팅하고, ddpm.* 은 ./mainddpm 으로 해소한다.
import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from torch.nn.functional import adaptive_avg_pool2d

# [EXP] 절대 import 로 재작성 (원본은 package 컨텍스트를 요구하는 `..` 상대 import).
from ddpm.models.ema import EMAHelper
from ddpm.functions import get_optimizer
from ddpm.functions.losses import loss_registry
from ddpm.datasets import get_dataset, data_transform, inverse_data_transform
from ddpm.functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu
from ddpm.utils import tools
logger = logging.getLogger(__name__)

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, interval_seq=None):
        self.args = args
        self.config = config
        self.accelerator = args.accelerator
        self.device = self.accelerator.device
        self.config.device = self.device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), alphas_cumprod[:-1]], dim=0
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        self.interval_seq = interval_seq

    def creat_model(self):
        if self.args.cache:
            from ddpm.models.deepcache_diffusion import Model
            model = Model(self.config)
            model.set_cache_para(self.args.branch)
            model._exp_backend = "deepcache"
            logger.info('Sampling in DeepCache mode')
        else:
            # [EXP-C4] fp16 / 양자화 off 비교용 non-cache 경로 (실험 5 Setting 1).
            from ddpm.models.diffusion import Model
            model = Model(self.config)
            model._exp_backend = "plain"
            logger.info('Sampling in non-DeepCache (fp16) mode')

        if self.config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif self.config.data.dataset == "LSUN":
            name = f"lsun_{self.config.data.category}"
        else:
            raise ValueError
        ckpt = get_ckpt_path(f"ema_{name}")
        logger.info("Loading checkpoint {}".format(ckpt))
        msg = model.load_state_dict(torch.load(ckpt, map_location=self.device), strict=False)

        logger.info(msg)
        model.cuda()
        model.eval()
        return model

    def sample(self):
        if self.args.cache:
            from ddpm.models.deepcache_diffusion import Model
            model = Model(self.config)
            logger.info('Sampling in DeepCache mode')
        else:
            from ddpm.models.diffusion import Model
            model = Model(self.config)
       
        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.logger.checkpoint_path, "ckpt.pth"),
                    map_location='cpu',
                )
                logger.info("Loading from latest checkpoint: {}".format(
                    os.path.join(self.logger.checkpoint_path, "ckpt.pth")
                ))
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location='cpu',
                )
                logger.info("Loading from latest checkpoint: {}".format(
                    os.path.join(self.logger.checkpoint_path, f"ckpt_{self.config.sampling.ckpt_id}.pth")
                ))
            model.load_state_dict(tools.unwrap_module(states[0]), strict=True)
            
            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(tools.unwrap_module(states[-1]))
                ema_helper.ema(model)
            else:
                ema_helper = None
            
            model = self.accelerator.prepare(model)
        else:
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            logger.info("Loading checkpoint {}".format(ckpt))
            msg = model.load_state_dict(torch.load(ckpt, map_location=self.device), strict=False)

            logger.info(msg)
            model = self.accelerator.prepare(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model, total_n_samples=self.args.max_images)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model, total_n_samples=50000, save_images = True, timesteps=None):
        config = self.config
        # img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        img_id = 0
        logger.info(f"starting from image {img_id}")
        total_n_samples = total_n_samples // self.accelerator.num_processes
        # n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        n_rounds = (total_n_samples - img_id) // self.args.sample_batch

        generate_samples = []
        throughput = []
        sample_start_time = time.time()
        with torch.no_grad(), tqdm.tqdm(range(n_rounds)) as t:
            
            for _ in t:
                start_time = time.time()
                # n = config.sampling.batch_size
                n = self.args.sample_batch
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model, timesteps=timesteps)
                x = inverse_data_transform(config, x)

                use_time = time.time() - start_time
                throughput.append(x.shape[0] / use_time)
                t.set_description(f"Throughput: {np.mean(throughput):.2f} samples/s")
                
                if save_images:
                    for i in range(n):
                        tvu.save_image(
                            x[i], os.path.join(self.args.image_folder, f"{self.accelerator.process_index}_{img_id}.png")
                        )
                        img_id += 1
                else:
                    generate_samples.append(x)
        
        self.args.accelerator.wait_for_everyone()
        logger.info(f"Time taken: {time.time() - sample_start_time} seconds")
        return generate_samples

    
    def sample_image(self, x, model, last=True, timesteps=None):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        if timesteps is None:
            timesteps = self.args.timesteps
        #print(self.args.sample_type, self.args.skip_type, timesteps)

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError

            correction_mode = getattr(self.args, "correction_mode", None)
            if not correction_mode or correction_mode == "auto":
                correction_mode = "siec" if getattr(self.args, "use_siec", False) else "iec"

            if self.interval_seq == None:
                # [EXP-C4] interval_seq 가 없으면 S-IEC lookahead 경로가 동작하지 않는다.
                # S-IEC 를 요구하는 run 이 이 분기로 들어오면 silent 하게 IEC 로 fallback 되지
                # 않도록 명시적으로 막는다. (fp16 S-IEC 는 별도 sampler 필요)
                if correction_mode not in ("none",):
                    raise NotImplementedError(
                        "[EXP] correction_mode requires interval_seq-backed sampling. "
                        "Use the experimental copy with DeepCache calibration loaded, or "
                        "switch to correction_mode=none."
                    )
                from deepcache_denoising import generalized_steps
                xs = generalized_steps(
                    x, seq, model, self.betas,
                    timesteps=timesteps,
                    cache_interval=getattr(self.args, 'cache_interval', 1),
                    non_uniform=self.args.non_uniform, pow=self.args.pow,
                    center=self.args.center, branch=self.args.branch,
                    eta=self.args.eta)
            else:
                common_kwargs = dict(
                    timesteps=timesteps,
                    interval_seq=self.interval_seq,
                    branch=self.args.branch,
                    eta=self.args.eta,
                    quant=self.args.ptq,
                    disable_cache_reuse=getattr(self.args, "disable_cache_reuse", False),
                )

                if getattr(self.args, 'siec_return_trace', False):
                    from deepcache_denoising import adaptive_generalized_steps_trace
                    trace_mode = getattr(self.args, "siec_trace_mode", "auto")
                    if trace_mode == "auto":
                        trace_mode = correction_mode
                    result = adaptive_generalized_steps_trace(
                        x, seq, model, self.betas,
                        mode=trace_mode,
                        c_siec=getattr(self.args, 'c_siec', 1.0),
                        tau_schedule=getattr(self.args, 'tau_schedule', None),
                        siec_always_correct=getattr(self.args, 'siec_always_correct', False),
                        siec_max_rounds=getattr(self.args, 'siec_max_rounds', 1),
                        trigger_mode=getattr(self.args, 'trigger_mode', 'syndrome'),
                        trigger_prob=getattr(self.args, 'trigger_prob', 0.2),
                        trigger_period=getattr(self.args, 'trigger_period', 5),
                        trace_include_x0=getattr(self.args, "trace_include_x0", False),
                        **common_kwargs,
                    )
                    xs_list, x0_preds_list, trace = result
                    if not hasattr(self.args, '_exp_traces'):
                        self.args._exp_traces = []
                    self.args._exp_traces.append(trace)
                    xs = (xs_list, x0_preds_list)
                elif correction_mode == "none":
                    from deepcache_denoising import adaptive_generalized_steps_none
                    xs = adaptive_generalized_steps_none(
                        x, seq, model, self.betas, **common_kwargs
                    )
                elif correction_mode == "iec":
                    from deepcache_denoising import adaptive_generalized_steps_3
                    xs = adaptive_generalized_steps_3(
                        x, seq, model, self.betas, **common_kwargs
                    )
                elif correction_mode == "siec":
                    from deepcache_denoising import adaptive_generalized_steps_siec
                    result = adaptive_generalized_steps_siec(
                        x, seq, model, self.betas,
                        c_siec=getattr(self.args, 'c_siec', 1.0),
                        tau_schedule=getattr(self.args, 'tau_schedule', None),
                        siec_always_correct=getattr(self.args, 'siec_always_correct', False),
                        siec_collect_scores=getattr(self.args, 'siec_collect_scores', False),
                        siec_max_rounds=getattr(self.args, 'siec_max_rounds', 1),
                        trigger_mode=getattr(self.args, 'trigger_mode', 'syndrome'),
                        trigger_prob=getattr(self.args, 'trigger_prob', 0.2),
                        trigger_period=getattr(self.args, 'trigger_period', 5),
                        **common_kwargs,
                    )
                    if getattr(self.args, 'siec_collect_scores', False) and len(result) == 3:
                        xs_list, x0_preds_list, batch_scores = result
                        if not hasattr(self.args, '_pilot_scores'):
                            self.args._pilot_scores = []
                        self.args._pilot_scores.append(batch_scores)
                        xs = (xs_list, x0_preds_list)
                    else:
                        xs = result
                elif correction_mode == "tac":
                    from deepcache_denoising import adaptive_generalized_steps_tac
                    xs = adaptive_generalized_steps_tac(
                        x, seq, model, self.betas, **common_kwargs
                    )
                else:
                    raise ValueError(f"unknown correction_mode: {correction_mode}")
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            # Not implemented for DeepCache
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from deepcache_denoising import ddpm_steps
            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x





        
