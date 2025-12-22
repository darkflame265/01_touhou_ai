def __init__(
        self,
        screen,
        weight_path="weights/reimu_heatmap_best.pt",
        beta=12.0,
        prior_strength=1.0,
        ema_alpha=0.75,
        device=None,

        # ===== tracking 옵션 =====
        track_prior_strength=2.0,
        track_prior_sigma=0.08,
        lock_conf_thr=0.015,
        max_jump_norm=0.22,
        jump_allow_conf_gain=1.8,
        lost_patience=8,

        # ===== 성능 옵션 =====
        use_fp16=True,            # CUDA에서만 의미 있음
        track_prior_every=2,       # 1=매프레임, 2=2프레임마다, 3=3프레임마다...
        print_prof=True,          # det 내부 프로파일 출력
        prof_every=200,            # 몇 step마다 출력할지
    ):