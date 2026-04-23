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
                    et, cur_f = model(xt, t, prv_f=None,branch=branch)
                    prv_f = cur_f
                    save_features.append(cur_f[0].detach().cpu())
                    slow_path_count+= 1
                else:
                    et, cur_f = model(xt, t, prv_f=prv_f,branch=branch)
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


def adaptive_generalized_steps_3(x, seq, model, b, timesteps, interval_seq=None, branch=None, quant=False, **kwargs):
    print('runing to function adaptive_generalized_steps_3')

    '''
    cur_i in interval_seq determines to whether the IEC is used
    '''
    model.timesteps = timesteps
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        prv_f = None
        cur_i = 0
        tmp_ets = []
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            # if quant:
            if True:
                time = len(xs) - 1
                if quant:
                    model.set_time(time) # 
                total_steps = len(seq)
                enable_implicit = (cur_i in interval_seq)
                # enable_implicit = False
                if enable_implicit:
                    max_iter = 2  
                else:
                    max_iter = 1  
                tol = 1e-3  # error threshold
                for iter in range(max_iter):  # 
                    # 
                    if iter == 0:
                        if cur_i in interval_seq:  # %
                            et, cur_f = model(xt, t, context=None, prv_f=None, branch=branch)
                            prv_f = cur_f[0]
                        else:
                            et, cur_f = model(xt, t, context=None, prv_f=prv_f, branch=branch)
                        if quant:
                            model.model.time = model.model.time - 1
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                        x0_preds.append(x0_t.to('cpu'))
                        c1 = (
                                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                        )
                        c2 = ((1 - at_next) - c1 ** 2).sqrt()
                        # 
                        xt_next_hat = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                        # xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                        # xs.append(xt_next.to('cpu'))
                        # cur_i += 1
                    else:
                        # use xt_next_hat as input
                        if cur_i in interval_seq:  # %
                            et, cur_f = model(xt_next_hat, t, context=None, prv_f=None,
                                              branch=branch)   
                            prv_f = cur_f[0]
                        else:
                            et, cur_f = model(xt_next_hat, t, context=None, prv_f=prv_f,
                                              branch=branch)  
                        if quant:
                            model.model.time = model.model.time - 1

                        # 隐式更新公式（对应理论方程 x_{t-1} = A_t x_t + B_t ε(x_{t-1})）
                        xt_next_new = at_next.sqrt() * ((xt - et * (1 - at).sqrt()) / at.sqrt()) + \
                                      c1 * torch.randn_like(x) + c2 * et

                        # chech if it is converged
                        # if torch.norm(xt_next_new - xt_next_hat) < tol:
                        residual = torch.norm(xt_next_new - xt_next_hat) / (torch.norm(xt_next_hat) + 1e-6)
                        if residual < tol:
                            break

                        
                        gamma = 0.5
                        xt_next_hat = xt_next_hat + (gamma ** iter) * (xt_next_new - xt_next_hat)
                if quant:
                    model.model.time = model.model.time + 1
                # take output
                xs.append(xt_next_hat.to('cpu'))
                cur_i += 1
            else:
                pass
                if cur_i in interval_seq:  # %
                    et, cur_f = model(xt, t, context=None, prv_f=None, branch=branch)
                    prv_f = cur_f[0]
                else:
                    et, cur_f = model(xt, t, context=None, prv_f=prv_f, branch=branch)
                # tmp_ets.append(et.detach().cpu().numpy())
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                x0_preds.append(x0_t.to('cpu'))
                c1 = (
                        kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                # print(c1, c2)
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next.to('cpu'))
                cur_i += 1

    return xs, x0_preds



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
    # S-IEC specific
    c_siec=1.0,
    tau_schedule=None,
    siec_always_correct=False,
    siec_collect_scores=False,
    siec_max_rounds=1,
    # [EXP-C1] 실험 4 트리거 모드 베이스라인 (random/uniform vs syndrome)
    trigger_mode="syndrome",
    trigger_prob=0.2,
    trigger_period=5,
    **kwargs
):
    """
    S-IEC: Syndrome-guided test-time Error Correction.
    
    Follows Prof's toy SIEC.correct_step structure as closely as possible,
    adapted for IEC's DeepCache + CacheQuant pipeline.
    
    Design choice: syndrome check only at interval_seq positions
    (DeepCache refresh points), to preserve IEC's compute structure.
    
    Correction loop mirrors toy SIEC.correct_step:
      for _ in range(max_rounds):
          x0_corrected = consensus_correction(...)
          xt_corrected = ddim_step(xt, x0_corrected, ...)
          x0_look_new = re_evaluate_lookahead(xt_corrected, t-1)
          if new_score <= tau: break
          x0_look = x0_look_new
    """
    
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

        # Diagnostics (for analysis / pilot)
        collected_scores = []   # score at each step (batch mean)
        triggered_flags = []    # whether correction triggered
        nfe_per_step = []       # NFE count per step

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            time = len(xs) - 1
            if quant:
                model.set_time(time)

            step_nfe = 0

            # ============================================
            # Step 1: Forward at t
            # ============================================
            if cur_i in interval_seq:
                et, cur_f = model(xt, t, context=None, prv_f=None, branch=branch)
                prv_f = cur_f[0]
            else:
                et, cur_f = model(xt, t, context=None, prv_f=prv_f, branch=branch)
            if quant:
                model.model.time = model.model.time - 1
            step_nfe += 1

            # x̂_0 at t
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            # Standard DDIM coefficients
            c1 = kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            # Save the noise used for c1 term (same noise reused in corrections)
            noise = torch.randn_like(x)

            # ============================================
            # Step 2: Tentative transition
            # ============================================
            xt_next_tent = at_next.sqrt() * x0_t + c1 * noise + c2 * et
            xt_next_hat = xt_next_tent

            # ============================================
            # Step 3~5: S-IEC syndrome check (only at interval_seq)
            # ============================================
            do_siec_check = (cur_i in interval_seq) and (next_t.long()[0].item() >= 0)
            batch_score_value = 0.0
            triggered = False

            if do_siec_check:
                # Lookahead forward at t-1 on tentative xt_next
                et_look, _ = model(xt_next_tent, next_t, context=None, prv_f=None, branch=branch)
                if quant:
                    model.model.time = model.model.time - 1
                step_nfe += 1

                x0_look = (xt_next_tent - et_look * (1 - at_next).sqrt()) / at_next.sqrt()

                # Syndrome: ŝ = x̂_0(t) - x̂_0(t-1)
                syndrome, score = compute_syndrome(x0_t, x0_look)
                batch_score_value = float(score.mean().item())

                if siec_collect_scores:
                    # Pilot mode: just record, don't correct
                    pass
                else:
                    # Threshold decision
                    if siec_always_correct:
                        triggered = True
                    # [EXP-C1] 베이스라인 트리거 모드. syndrome 체크 게이트 안에 두어
                    # step 별 NFE bookkeeping 이 모드 간에 비교 가능하게 유지.
                    elif trigger_mode == "random":
                        triggered = bool(np.random.random() < trigger_prob)
                    elif trigger_mode == "uniform":
                        triggered = (cur_i % max(1, trigger_period) == 0)
                    elif tau_schedule is not None:
                        tau_t = float(tau_schedule[cur_i]) if cur_i < len(tau_schedule) else 0.0
                        triggered = (batch_score_value > tau_t)

                    if triggered:
                        # ============================================
                        # Step 5: Correction loop (matches toy correct_step)
                        # ============================================
                        alpha_t_sq = float(at.reshape(-1)[0].item())
                        sigma_t_sq = 1.0 - alpha_t_sq
                        gamma = compute_gamma(alpha_t_sq, sigma_t_sq, c=c_siec)
                        
                        for _round in range(siec_max_rounds):
                            x0_corrected = apply_consensus_correction(x0_t, syndrome, gamma)
                            # ★ 핵심 수정: ε를 x0_corrected와 일관되게 재유도
                            et_corrected = (xt - at.sqrt() * x0_corrected) / (1 - at).sqrt()
                            xt_next_hat = at_next.sqrt() * x0_corrected + c1 * noise + c2 * et_corrected

                            # For max_rounds > 1: re-evaluate lookahead
                            if _round < siec_max_rounds - 1:
                                et_look_new, _ = model(
                                    xt_next_hat, next_t, context=None,
                                    prv_f=None, branch=branch
                                )
                                if quant:
                                    model.model.time = model.model.time - 1
                                step_nfe += 1

                                x0_look_new = (
                                    xt_next_hat - et_look_new * (1 - at_next).sqrt()
                                ) / at_next.sqrt()
                                syndrome, new_score = compute_syndrome(
                                    x0_t, x0_look_new
                                )
                                new_score_val = float(new_score.mean().item())

                                # Early termination check
                                if tau_schedule is not None:
                                    tau_t = float(tau_schedule[cur_i]) if cur_i < len(tau_schedule) else 0.0
                                    if new_score_val <= tau_t:
                                        break

            collected_scores.append(batch_score_value)
            triggered_flags.append(triggered)
            nfe_per_step.append(step_nfe)

            xs.append(xt_next_hat.to('cpu'))
            cur_i += 1

    # Return diagnostics (attached via special flag or structure)
    if siec_collect_scores:
        return xs, x0_preds, collected_scores
    
    # For non-pilot: return standard tuple
    # Trigger/NFE info stored on xs as attribute (optional, for debugging)
    return xs, x0_preds


def adaptive_generalized_steps_trace(
    x, seq, model, b, timesteps,
    interval_seq=None, branch=None, quant=False,
    mode='iec',   # 'iec' or 'siec'
    # S-IEC options (mode='siec'일 때만)
    c_siec=1.0,
    tau_schedule=None,
    siec_always_correct=False,
    siec_max_rounds=1,
    **kwargs
):
    """
    Path error / martingale deviation 측정 전용 sampling.
    
    IEC와 S-IEC의 x0 trajectory를 기록해서 비교할 수 있게 함.
    
    Returns:
        xs, x0_preds: 기존과 동일 (sample_image가 쓸 수 있도록)
        trace: dict with diagnostic arrays
            - x0_trajectory: list of (B, C, H, W) per reverse step
            - syndrome_per_step: list of float (batch mean r_t)
            - triggered_per_step: list of bool (correction 여부)
            - nfe_per_step: list of int
    """
    
    from siec_core.syndrome import compute_syndrome
    if mode == 'siec':
        from siec_core.correction import compute_gamma, apply_consensus_correction

    model.timesteps = timesteps
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        prv_f = None
        cur_i = 0
        
        # ★ Trace 기록용
        x0_trajectory = []        # 모든 reverse step의 x0 추정
        syndrome_per_step = []    # syndrome score (IEC는 항상 계산, S-IEC는 자연히)
        triggered_per_step = []
        nfe_per_step = []
        
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            
            time_idx = len(xs) - 1
            if quant:
                model.set_time(time_idx)
            
            step_nfe = 0
            
            # Forward at t
            if cur_i in interval_seq:
                et, cur_f = model(xt, t, context=None, prv_f=None, branch=branch)
                prv_f = cur_f[0]
            else:
                et, cur_f = model(xt, t, context=None, prv_f=prv_f, branch=branch)
            if quant:
                model.model.time = model.model.time - 1
            step_nfe += 1
            
            # x̂_0(t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            x0_trajectory.append(x0_t.detach().cpu().clone())   # ★ 기록
            
            # DDIM coefficients
            c1 = kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            noise = torch.randn_like(x)
            
            # Tentative
            xt_next_tent = at_next.sqrt() * x0_t + c1 * noise + c2 * et
            xt_next_hat = xt_next_tent
            
            # ============================================
            # Syndrome 측정 (두 모드 다 측정, correction은 mode별)
            # ============================================
            batch_score = 0.0
            triggered = False
            
            do_lookahead = (cur_i in interval_seq) and (next_t.long()[0].item() >= 0)
            
            if do_lookahead:
                # lookahead forward
                et_look, _ = model(xt_next_tent, next_t, context=None, prv_f=None, branch=branch)
                if quant:
                    model.model.time = model.model.time - 1
                step_nfe += 1
                
                x0_look = (xt_next_tent - et_look * (1 - at_next).sqrt()) / at_next.sqrt()
                
                # ★ syndrome 계산 (IEC도 측정, 비교용)
                syndrome, score_per_sample = compute_syndrome(x0_t, x0_look)
                batch_score = float(score_per_sample.mean().item())
                
                # ============================================
                # Mode별 분기
                # ============================================
                if mode == 'iec':
                    # IEC: fixed-point iteration (max_iter=2)
                    # 원본 IEC 로직 재현
                    for it in range(1, 2):   # iter 1 (이미 iter 0는 했음)
                        # forward at t (xt_next_hat 입력)
                        if cur_i in interval_seq:
                            et_new, cur_f_new = model(xt_next_hat, t, context=None, prv_f=None, branch=branch)
                            prv_f = cur_f_new[0]
                        else:
                            et_new, cur_f_new = model(xt_next_hat, t, context=None, prv_f=prv_f, branch=branch)
                        if quant:
                            model.model.time = model.model.time - 1
                        step_nfe += 1
                        
                        xt_next_new = at_next.sqrt() * ((xt - et_new * (1 - at).sqrt()) / at.sqrt()) + \
                                      c1 * noise + c2 * et_new
                        
                        residual = torch.norm(xt_next_new - xt_next_hat) / (torch.norm(xt_next_hat) + 1e-6)
                        if residual < 1e-3:
                            break
                        
                        gamma_iter = 0.5
                        xt_next_hat = xt_next_hat + (gamma_iter ** it) * (xt_next_new - xt_next_hat)
                    
                    triggered = True   # IEC는 interval_seq 위치에서 항상 iteration
                
                elif mode == 'siec':
                    # S-IEC: threshold-based trigger
                    if siec_always_correct:
                        triggered = True
                    elif tau_schedule is not None:
                        tau_t = float(tau_schedule[cur_i]) if cur_i < len(tau_schedule) else 0.0
                        triggered = (batch_score > tau_t)
                    
                    if triggered:
                        alpha_t_sq = float(at.reshape(-1)[0].item())
                        sigma_t_sq = 1.0 - alpha_t_sq
                        gamma = compute_gamma(alpha_t_sq, sigma_t_sq, c=c_siec)
                        
                        for _round in range(siec_max_rounds):
                            x0_corrected = apply_consensus_correction(x0_t, syndrome, gamma)
                            # ★ 핵심 수정
                            et_corrected = (xt - at.sqrt() * x0_corrected) / (1 - at).sqrt()
                            xt_next_hat = at_next.sqrt() * x0_corrected + c1 * noise + c2 * et_corrected
        
                            if _round < siec_max_rounds - 1:
                                # 재검증 (max_rounds > 1일 때만)
                                et_look_new, _ = model(xt_next_hat, next_t, context=None,
                                                       prv_f=None, branch=branch)
                                if quant:
                                    model.model.time = model.model.time - 1
                                step_nfe += 1
                                x0_look_new = (xt_next_hat - et_look_new * (1 - at_next).sqrt()) / at_next.sqrt()
                                syndrome, new_score = compute_syndrome(x0_t, x0_look_new)
                                new_score_val = float(new_score.mean().item())
                                if tau_schedule is not None and new_score_val <= tau_t:
                                    break
            
            # non interval_seq 위치: 그냥 DDIM (baseline iec/no-correction)
            # no-op, xt_next_hat은 이미 tentative
            
            syndrome_per_step.append(batch_score)
            triggered_per_step.append(triggered)
            nfe_per_step.append(step_nfe)
            
            xs.append(xt_next_hat.to('cpu'))
            cur_i += 1
    
    trace = {
        'x0_trajectory': x0_trajectory,       # list of CPU tensors
        'syndrome_per_step': syndrome_per_step,
        'triggered_per_step': triggered_per_step,
        'nfe_per_step': nfe_per_step,
    }
    
    return xs, x0_preds, trace
