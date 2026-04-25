# [EXP] /home/user/jowithu/Semantic/IEC/mainddpm/ddpm/functions/deepcache_denoising.py 의 실험용 복사본.
# 원본은 수정하지 않음. 아래 수정 부분은 # [EXP-CN] 태그로 표시.
# 엔트리인 ddim_cifar_siec.py 의 sys.path shim 이 이 실험 내에서만 원본을 가린다.
import torch

from scipy.stats import shapiro
import numpy as np

def sample_gaussian_centered(n=1000, sample_size=100, std_dev=100, shift=0):
    samples = []
    
    while len(samples) < sample_size:
        # Sample from a Gaussian centered at n/2
        sample = int(np.random.normal(loc=n/2+shift, scale=std_dev))
        
        # Check if the sample is in bounds
        if 1 <= sample < n and sample not in samples:
            samples.append(sample)
    
    return samples

def sample_from_quad_center(total_numbers, n_samples, center, pow=1.2):
    while pow > 1:
        # Generate linearly spaced values between 0 and a max value
        x_values = np.linspace((-center)**(1/pow), (total_numbers-center)**(1/pow), n_samples+1)
        #print(x_values)
        #print([x for x in np.unique(np.int32(x_values**pow))[:-1]])
        # Raise these values to the power of 1.5 to get a non-linear distribution
        indices = [0] + [x+center for x in np.unique(np.int32(x_values**pow))[1:-1]]
        if len(indices) == n_samples:
            break
        
        pow -=0.02
    return indices, pow

def sample_from_quad(total_numbers, n_samples, pow=1.2):
    # Generate linearly spaced values between 0 and a max value
    x_values = np.linspace(0, total_numbers**(1/pow), n_samples+1)

    # Raise these values to the power of 1.5 to get a non-linear distribution
    indices = np.unique(np.int32(x_values**pow))[:-1]
    return indices

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, timesteps, cache_interval=None, non_uniform=False, pow=None, center=None,  branch=None, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        prv_f = None

        cur_i = 0
        if non_uniform:
            num_slow = timesteps // cache_interval
            if timesteps % cache_interval > 0:
                num_slow += 1
            interval_seq, final_pow = sample_from_quad_center(total_numbers=timesteps, n_samples=num_slow, center=center, pow=pow)
        else:
            interval_seq = list(range(0, timesteps, cache_interval))
            interval = cache_interval
        #print(non_uniform, interval_seq)
        

        slow_path_count = 0
        save_features = []
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            with torch.no_grad():
                if cur_i in interval_seq: #%
                #if cur_i % interval == 0:
                    #print(cur_i, interval_seq)
                    et, cur_f = _call_model(
                        model,
                        xt,
                        t,
                        branch=branch,
                        prv_f=None,
                        allow_cache_reuse=False,
                    )
                    prv_f = cur_f
                    if isinstance(cur_f, (list, tuple)) and cur_f:
                        save_features.append(cur_f[0].detach().cpu())
                    elif cur_f is not None and hasattr(cur_f, "detach"):
                        save_features.append(cur_f.detach().cpu())
                    slow_path_count+= 1
                else:
                    et, cur_f = _call_model(
                        model,
                        xt,
                        t,
                        branch=branch,
                        prv_f=prv_f,
                        allow_cache_reuse=True,
                    )
                    #quick_path_count+= 1

            #print(i, torch.mean(et) / torch.mean(xt), torch.var(et)/torch.var(xt), torch.mean(et), torch.var(et))

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

            cur_i += 1

    return xs, x0_preds


def compute_spectral_radius(model, xt, t, b, branch=None, power_iter=3,):
    """
    计算当前时间步雅可比矩阵的谱半径 (ρ(J_f))

    参数:
        model: 去噪模型 εθ(x_t, t)
        xt: 当前状态张量 (shape: [B,C,H,W])
        t: 当前时间步张量 (shape: [B])
        branch:
        power_iter: 幂迭代次数

    返回:
        瑞利商 rho_J: 谱半径估计值 (标量)
        对于一般矩阵（包括非埃尔米特矩阵），谱半径𝜌(𝐴)ρ(A) 是矩阵特征值的最大模值。
        瑞利商本身不直接等于谱半径，但通过幂迭代法，瑞利商的绝对值可以逼近谱半径。
    """
    # ====== 数学背景 ======
    # 需要计算 ODE 右项函数 f(x,t) = (x - sqrt(1-α_t)εθ)/sqrt(α_t) - x
    # 雅可比矩阵 J_f = ∂f/∂x = (I/sqrt(α_t) - sqrt(1-α_t)/sqrt(α_t) ∂εθ/∂x) - I
    # 谱半径 ρ(J_f) = max|λ(J_f)|

    B, C, H, W = xt.shape
    xt = xt.detach().requires_grad_(True)  # 启用梯度跟踪

    # ====== 前向计算 f(x,t) ======
    # ====== 前向计算分离计算图 ======
    with torch.no_grad():
        et, _ = model(xt, t, context=None, prv_f=None, branch=branch)
    alpha_t = compute_alpha(b, t.long())

    # ====== 只包装需要微分的操作 ======
    with torch.enable_grad():
        et = et.detach().requires_grad_(True)  # 重新附加梯度
        f = (xt - (1 - alpha_t).sqrt() * et) / alpha_t.sqrt() - xt

    # ====== 幂迭代法 ======
    v = torch.randn_like(xt)  # 初始化随机向量
    v = v / (v.norm() + 1e-6)  # 单位化
    rho_J = 0.0
    for iter in range(power_iter):
        # 最后一次迭代关闭retain_graph
        retain = iter < (power_iter - 1)
        Jv = torch.autograd.grad(
            outputs=f,
            inputs=xt,
            grad_outputs=v,
            retain_graph=retain,  # 关键修改
            create_graph=False,
            only_inputs=True
        )[0]

        # 更新向量
        v = Jv.detach()
        v_norm = v.norm() + 1e-6
        v = v / v_norm

        # 计算瑞利商
        rho_J = (v * Jv).sum().abs().item()

        # 及时释放中间变量
        del Jv
        torch.cuda.empty_cache()

    # ====== 清理计算图 ======
    xt.grad = None
    del f, et
    torch.cuda.empty_cache()

    return rho_J

def find_topk_indices(lst, k):
    """
    找到列表中值最大的 Top-K 个元素的索引
    :param lst: 输入列表
    :param k: 需要找到的最大值的数量
    :return: 最大值对应的索引列表
    """
    import heapq
    # 使用 heapq.nlargest 找到值最大的 K 个元素及其索引
    topk = heapq.nlargest(k, enumerate(lst), key=lambda x: x[1])
    # 返回对应的索引
    return [index for index, value in topk]


def _correction_backend(model):
    backend = getattr(model, "_exp_backend", None)
    if backend:
        return backend
    inner = getattr(model, "model", None)
    if inner is not None:
        backend = getattr(inner, "_exp_backend", None)
        if backend:
            return backend
        if hasattr(inner, "set_cache_para"):
            return "deepcache"
    if hasattr(model, "set_cache_para"):
        return "deepcache"
    return "plain"


def _post_forward_quant_step(model, quant):
    if not quant:
        return
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "time"):
        inner.time = inner.time - 1


def _restore_quant_step(model, quant):
    if not quant:
        return
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "time"):
        inner.time = inner.time + 1


def _next_cache_feature(cur_f):
    if isinstance(cur_f, (list, tuple)) and cur_f:
        return cur_f[0]
    return cur_f


def _call_model(model, xt, t, branch=None, prv_f=None, allow_cache_reuse=True):
    backend = _correction_backend(model)
    if backend == "deepcache":
        use_prv = prv_f if allow_cache_reuse else None
        et, cur_f = model(xt, t, context=None, prv_f=use_prv, branch=branch)
        return et, _next_cache_feature(cur_f)
    try:
        et = model(xt, t)
    except TypeError:
        et = model(xt, t, context=None)
    return et, None


def _is_interval_step(cur_i, interval_seq):
    return interval_seq is not None and cur_i in interval_seq


def _ddim_transition(x_ref, at, at_next, x0_t, et, noise, eta):
    c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    xt_next = at_next.sqrt() * x0_t + c1 * noise + c2 * et
    return xt_next, c1, c2


def _adaptive_generalized_core(
    x, seq, model, b, timesteps,
    interval_seq=None, branch=None, quant=False,
    correction_mode="none",
    c_siec=1.0,
    tau_schedule=None,
    siec_always_correct=False,
    siec_collect_scores=False,
    siec_max_rounds=1,
    trigger_mode="syndrome",
    trigger_prob=0.2,
    trigger_period=5,
    disable_cache_reuse=False,
    collect_trace=False,
    trace_include_x0=False,
    **kwargs
):
    if correction_mode == "tac":
        raise NotImplementedError("TAC correction is not implemented in this repository")

    if correction_mode == "siec":
        from siec_core.syndrome import compute_syndrome
        from siec_core.correction import compute_gamma, apply_consensus_correction

    model.timesteps = timesteps
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        prv_f = None
        cur_i = 0

        x0_trajectory = []
        syndrome_per_step = []
        score_values_per_step = []
        triggered_per_step = []
        checked_per_step = []
        nfe_per_step = []
        collected_scores = []

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to("cuda")

            time_idx = len(xs) - 1
            if quant and hasattr(model, "set_time"):
                model.set_time(time_idx)

            refresh_step = _is_interval_step(cur_i, interval_seq)
            allow_cache_reuse = (not disable_cache_reuse) and (not refresh_step)

            step_nfe = 0
            et, next_prv_f = _call_model(
                model,
                xt,
                t,
                branch=branch,
                prv_f=prv_f,
                allow_cache_reuse=allow_cache_reuse,
            )
            _post_forward_quant_step(model, quant)
            step_nfe += 1

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to("cpu"))
            if trace_include_x0:
                x0_trajectory.append(x0_t.detach().cpu().clone())

            noise = torch.randn_like(x)
            xt_next_hat, c1, c2 = _ddim_transition(
                x_ref=x,
                at=at,
                at_next=at_next,
                x0_t=x0_t,
                et=et,
                noise=noise,
                eta=kwargs.get("eta", 0),
            )

            checked = False
            triggered = False
            batch_score = 0.0
            score_values = []

            if correction_mode == "iec" and refresh_step:
                checked = True
                et_new, next_prv_candidate = _call_model(
                    model,
                    xt_next_hat,
                    t,
                    branch=branch,
                    prv_f=prv_f,
                    allow_cache_reuse=False,
                )
                _post_forward_quant_step(model, quant)
                step_nfe += 1
                xt_next_new, _c1, _c2 = _ddim_transition(
                    x_ref=x,
                    at=at,
                    at_next=at_next,
                    x0_t=(xt - et_new * (1 - at).sqrt()) / at.sqrt(),
                    et=et_new,
                    noise=noise,
                    eta=kwargs.get("eta", 0),
                )
                residual = torch.norm(xt_next_new - xt_next_hat) / (torch.norm(xt_next_hat) + 1e-6)
                xt_next_hat = xt_next_new if residual < 1e-3 else xt_next_hat + 0.5 * (xt_next_new - xt_next_hat)
                _restore_quant_step(model, quant)
                if next_prv_candidate is not None:
                    next_prv_f = next_prv_candidate
                triggered = True

            if correction_mode == "siec":
                # Upstream fix: syndrome check is done at every reverse timestep.
                # DeepCache refresh points still control cache reuse on the forward
                # path, but they must not gate whether S-IEC evaluates/corrects.
                checked = (next_t.long()[0].item() >= 0)
                if checked:
                    et_look, _ = _call_model(
                        model,
                        xt_next_hat,
                        next_t,
                        branch=branch,
                        prv_f=None,
                        allow_cache_reuse=False,
                    )
                    _post_forward_quant_step(model, quant)
                    step_nfe += 1

                    x0_look = (xt_next_hat - et_look * (1 - at_next).sqrt()) / at_next.sqrt()
                    syndrome, score = compute_syndrome(x0_t, x0_look)
                    batch_score = float(score.mean().item())
                    score_values = [float(v) for v in score.detach().cpu().reshape(-1).tolist()]

                    if siec_collect_scores:
                        # Preserve the per-sample score distribution for tau
                        # calibration; batch means are too coarse.
                        collected_scores.append(score_values)
                    else:
                        if siec_always_correct:
                            triggered = True
                        elif trigger_mode == "random":
                            triggered = bool(np.random.random() < trigger_prob)
                        elif trigger_mode == "uniform":
                            triggered = (cur_i % max(1, trigger_period) == 0)
                        elif tau_schedule is not None:
                            tau_t = float(tau_schedule[cur_i]) if cur_i < len(tau_schedule) else 0.0
                            triggered = batch_score > tau_t

                        if triggered:
                            alpha_t_sq = float(at.reshape(-1)[0].item())
                            sigma_t_sq = 1.0 - alpha_t_sq
                            gamma = compute_gamma(alpha_t_sq, sigma_t_sq, c=c_siec)

                            for round_idx in range(max(1, siec_max_rounds)):
                                x0_corrected = apply_consensus_correction(x0_t, syndrome, gamma)
                                et_corrected = (xt - at.sqrt() * x0_corrected) / (1 - at).sqrt()
                                xt_next_hat = at_next.sqrt() * x0_corrected + c1 * noise + c2 * et_corrected

                                if round_idx < max(1, siec_max_rounds) - 1:
                                    et_look_new, _ = _call_model(
                                        model,
                                        xt_next_hat,
                                        next_t,
                                        branch=branch,
                                        prv_f=None,
                                        allow_cache_reuse=False,
                                    )
                                    _post_forward_quant_step(model, quant)
                                    step_nfe += 1
                                    x0_look_new = (
                                        xt_next_hat - et_look_new * (1 - at_next).sqrt()
                                    ) / at_next.sqrt()
                                    syndrome, new_score = compute_syndrome(x0_t, x0_look_new)
                                    new_score_val = float(new_score.mean().item())
                                    if tau_schedule is not None:
                                        tau_t = float(tau_schedule[cur_i]) if cur_i < len(tau_schedule) else 0.0
                                        if new_score_val <= tau_t:
                                            break

            syndrome_per_step.append(batch_score)
            score_values_per_step.append(score_values)
            triggered_per_step.append(triggered)
            checked_per_step.append(checked)
            nfe_per_step.append(step_nfe)

            xs.append(xt_next_hat.to("cpu"))
            prv_f = next_prv_f
            cur_i += 1

    if siec_collect_scores:
        return xs, x0_preds, collected_scores

    if not collect_trace:
        return xs, x0_preds

    trace = {
        "batch_size": n,
        "correction_mode": correction_mode,
        "x0_trajectory": x0_trajectory,
        "syndrome_per_step": syndrome_per_step,
        "score_values_per_step": score_values_per_step,
        "triggered_per_step": triggered_per_step,
        "checked_per_step": checked_per_step,
        "nfe_per_step": nfe_per_step,
    }
    return xs, x0_preds, trace


def adaptive_generalized_steps_none(
    x, seq, model, b, timesteps, interval_seq=None, branch=None, quant=False, **kwargs
):
    return _adaptive_generalized_core(
        x,
        seq,
        model,
        b,
        timesteps,
        interval_seq=interval_seq,
        branch=branch,
        quant=quant,
        correction_mode="none",
        **kwargs,
    )


def adaptive_generalized_steps_3(
    x, seq, model, b, timesteps, interval_seq=None, branch=None, quant=False, **kwargs
):
    return _adaptive_generalized_core(
        x,
        seq,
        model,
        b,
        timesteps,
        interval_seq=interval_seq,
        branch=branch,
        quant=quant,
        correction_mode="iec",
        **kwargs,
    )



def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds


def adaptive_generalized_steps_siec(
    x, seq, model, b, timesteps,
    interval_seq=None, branch=None, quant=False,
    c_siec=1.0,
    tau_schedule=None,
    siec_always_correct=False,
    siec_collect_scores=False,
    siec_max_rounds=1,
    trigger_mode="syndrome",
    trigger_prob=0.2,
    trigger_period=5,
    **kwargs
):
    return _adaptive_generalized_core(
        x,
        seq,
        model,
        b,
        timesteps,
        interval_seq=interval_seq,
        branch=branch,
        quant=quant,
        correction_mode="siec",
        c_siec=c_siec,
        tau_schedule=tau_schedule,
        siec_always_correct=siec_always_correct,
        siec_collect_scores=siec_collect_scores,
        siec_max_rounds=siec_max_rounds,
        trigger_mode=trigger_mode,
        trigger_prob=trigger_prob,
        trigger_period=trigger_period,
        **kwargs,
    )


def adaptive_generalized_steps_tac(
    x, seq, model, b, timesteps, interval_seq=None, branch=None, quant=False, **kwargs
):
    return _adaptive_generalized_core(
        x,
        seq,
        model,
        b,
        timesteps,
        interval_seq=interval_seq,
        branch=branch,
        quant=quant,
        correction_mode="tac",
        **kwargs,
    )


def adaptive_generalized_steps_trace(
    x, seq, model, b, timesteps,
    interval_seq=None, branch=None, quant=False,
    mode="iec",
    c_siec=1.0,
    tau_schedule=None,
    siec_always_correct=False,
    siec_max_rounds=1,
    trigger_mode="syndrome",
    trigger_prob=0.2,
    trigger_period=5,
    trace_include_x0=False,
    **kwargs
):
    return _adaptive_generalized_core(
        x,
        seq,
        model,
        b,
        timesteps,
        interval_seq=interval_seq,
        branch=branch,
        quant=quant,
        correction_mode=mode,
        c_siec=c_siec,
        tau_schedule=tau_schedule,
        siec_always_correct=siec_always_correct,
        siec_collect_scores=False,
        siec_max_rounds=siec_max_rounds,
        trigger_mode=trigger_mode,
        trigger_prob=trigger_prob,
        trigger_period=trigger_period,
        collect_trace=True,
        trace_include_x0=trace_include_x0,
        **kwargs,
    )
