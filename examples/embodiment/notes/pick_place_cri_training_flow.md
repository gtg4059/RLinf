# Pick-place cube→bowl: 학습 흐름과 다음 CRI 반영용 기록

날짜: 2026-09-08  
목적: 지금 돌아가는 **성공 전용 PPO Ablation**의 루프를 고정하고, 이후 CRI 페널티를 다시 켤 때 같은 실수를 반복하지 않기 위함.  
수치 원본: `examples/embodiment/notes/pick_place_cri_runs.yaml`

## 1. 한 스텝의 학습 흐름

엔트리: `python examples/embodiment/train_embodied_agent.py --config-name isaaclab_pick_place_cube_plate_ppo_openpi_pi05_cri`  
러너: `rlinf/runners/embodied_runner.py` (`generate_rollouts` → `cal_adv_and_returns` → `actor_training`)

```
OpenPI 49999 가중치
        │
        ▼
[1] actor → rollout 가중치 동기화          weight_sync_interval: 1
        │
        ▼
[2] Rollout 예측 (HuggingFace / OpenPI flow)
        │  noise_level, num_action_chunks=15
        ▼
[3] Env.step (IsaacLab, 512 train / 256 eval)
        │  CRI는 run_cri_filter로 계산·로그만 (지금은 보상 0)
        │  성공/드롭에서 terminate
        │  auto_reset=False → hold-after-done
        ▼
[4] leftover 스텝: 마지막 abs-joint 고정, reward 0, 전 env done이면 physics skip
        │  다음 global step의 bootstrap_step()에서 reset
        ▼
[5] Actor가 trajectory 수신
        │  rewards = 청크 합 (성공 +1 한 번, CRI=0이면 그게 전부)
        ▼
[6] Advantage / return
        │  지금: GAE + normalize_advantages
        │  과거 CRI 실험: raw 또는 GAE(norm off)
        ▼
[7] PPO 업데이트 (FSDP, train_expert_only)
        │  지금: actor_critic, update_epoch=3, lr=5e-7
        ▼
[8] 메트릭 / (50스텝마다) eval·ckpt
```

관련 파일:

| 단계 | 파일 |
|---|---|
| 설정 | `examples/embodiment/config/isaaclab_pick_place_cube_plate_ppo_openpi_pi05_cri.yaml` |
| 비교용 성공 PPO | `examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml` |
| Env + CRI 보상 | `rlinf/envs/isaaclab/tasks/pick_place_cube_plate/env.py` |
| Hold | `.../pick_place_cube_plate/hold.py` |
| CRI solver | `.../pick_place_cube_plate/cri/solver.py` |
| Advantage | `rlinf/algorithms/advantages.py`, `utils.py` |
| PPO loss | `rlinf/algorithms/losses.py` |
| Actor | `rlinf/workers/actor/fsdp_actor_worker.py` |
| Critic warmup freeze | `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py` |

보상 식 (`env.py`):

```
r = r_task                          # sparse success, 보통 +1 한 번
if cri_penalty_weight != 0:
    r = r_task + weight * ovf(cri, limit=0.96, sigma)
```

`weight=0`이면 CRI는 `cri_max` 로그만 하고 PPO에 안 들어간다.

## 2. 지금 Ablation 설정 (성공만)

스택 큐브 PPO와 같은 알고리즘, 태스크/하드웨어만 pick-place.

- 보상: `cri_penalty_weight: 0`, 필터 OFF
- `adv_type: gae`, `loss_type: actor_critic`, `normalize_advantages: True`
- `update_epoch: 3`, `kl_beta: 0`, clip 0.2, `noise_level: 0.5`
- `lr: 5e-7`, `critic_warmup_steps: 0`
- Train: 512 env, `auto_reset: False`, `ignore_terminations: False`, horizon 450
- Eval: 256 env, `auto_reset: True`, `ignore_terminations: True`
- `resume_dir: null` (OpenPI 49999에서 시작)
- 런: `logs/20260908-07:47:07-...`  (step 9까지 train 성공 89.4%→86.0%)

이 런이 **유지·상승**해야 CRI를 다시 켤 자격이 있다.  
여기서도 내려가면 원인은 CRI가 아니라 “90% VLA + 성공/실패 PPO”다.

## 3. 인프라에서 이미 확정된 것

다시 바꾸지 말 것.

1. **Train `auto_reset: False` + `ignore_terminations: False`**  
   성공에서 에피소드 종료, leftover는 hold. `auto_reset: True`는 리셋 폭풍(~수천 reset/step)과 물리 폭주.
2. **성공했다고 리셋해서 CRI horizon을 줄이지 말 것.** Hold는 “성공 후 450까지 물리”가 아니라 “남은 스텝을 얼리고 보상 0”.
3. **액션은 CPU가 정상.** Env 마스크만 GPU. CPU action을 GPU 인덱스로 자르면 hold에서 죽음 (`hold.py`는 디바이스 맞춤 필요).
4. **`save_interval % val_check_interval == 0`.** 20과 50 조합은 assertion.
5. **8 GPU 배치:** train 512, eval 256, `global_batch_size` 10240, micro 128. rank당 롤아웃 3840이 1280으로 나눠떨어져야 함.
6. **`cri_filter_enabled`는 커리큘럼 마지막.** 성공이 유지되고 CRI가 내려간 뒤에만 ON.
7. **언러닝 스냅샷에서 resume 금지.** OpenPI `pi05_droid_cri_rlinf_49999` 또는 성공이 정점인 옛 `global_step_200`만.
8. **Eval CRI 가중치를 train과 맞출 것.** 기본 env yaml은 eval `-0.02` / σ 20이라 return/ovf가 비교 불가.
9. **`loss_type: actor` + critic warmup freeze는 `backward()` 크래시.**  
   Warmup이 actor를 `requires_grad=False`로 잠근다. actor-only면 그래프가 없다.  
   고침: actor-only일 때 freeze 생략 (`fsdp_model_manager.py`), loss는 `* 0.0`으로 그래프 유지 (`losses.py`).
10. Train `cri_ovf`는 hold 때문에 **마지막 스텝 값**에 가깝다. 에피소드 피크는 `cri_max`. 둘을 혼동하지 말 것.

## 4. 실험에서 나온 학습 결론

공통: OpenPI 시작 성공 ~90–93%. Train은 1024 traj, `episode_len=450`이면 hold는 동작 중.

| 결론 | 근거 |
|---|---|
| 작은 lr·강한 KL·긴 warmup만으로는 언러닝이 안 멈춤 | `14:07:05`: warmup 30, lr 5e-8 → actor ON 후 87%→76% (34스텝, −12%p). `06:24:28`과 기울기 동일 |
| CRI `-1e-4`~`-1e-3`는 +1 성공 옆에서 안 보임 | 청크 return 최댓값 ≈ 1/15=0.067. CRI 합은 그보다 한 자릿수 작음 |
| `normalize_advantages: True` + dense CRI는 CRI가 메인 신호가 됨 | 초기 붕괴 런 (성공 한 번, 450스텝 CRI) |
| GAE + 나쁜 critic는 advantage를 ±4~5로 부풂 | actor ON 순간 EV −15~−33 |
| Critic를 빼도(`raw`/`actor`) 성공은 내려감 | `05:57:42`: 91.5%→88.3% / 5 actor 스텝. grad_norm 0.3인데도 하락 |
| `noise_level: 0.1`은 FT 판단을 흐림 | lr=0인데 approx_kl 0.02, clip 25% |
| Eval 성공은 train보다 늦게 떨어짐 | `14:07:05` step 40: train 76%, eval 90.6% |
| 실패 advantage 절댓값이 성공보다 큼 | raw 기준 성공 +0.05, 실패 −0.17 → “실패하지 마라”가 사전학습을 깎음 |

다음 CRI에서 PPO가 봐야 하는 것: **이미 성공한 궤적의 `cri > 0.96`만**. 실패 vs 성공으로 정책을 밀면 다시 언러닝.

## 5. 다음 CRI를 켤 때 체크리스트

성공 Ablation이 88% 이상에서 버틴 뒤에만.

1. `cri_penalty_weight`를 0에서 올리기. 시작 후보 `-1e-3` (성공 전 ~150 라이브 스텝, 평균 ovf 0.3 → 합 약 -0.045, 성공 +1보다 훨씬 작음).
2. `normalize_advantages: False`로 둘 것. True면 CRI가 다시 주신호가 된다.
3. Train/eval `init_params` 가중치·sigma를 같게.
4. 필터는 끄고 시작. 성공 유지 + `cri_max` 하락이 보이면 `cri_filter_enabled: True`.
5. 가능하면 코드: 실패 궤적 advantage=0, `cri>0.96` 스텝만 약한 residual.
6. 워밍업을 길게 하는 것만으로 해결되지 않음. 빠른 FT를 보려면 warmup을 낮게 둬도 되지만, 그때 GAE+랜덤 critic을 같이 쓰지 말 것.
7. 중단 기준: train 성공이 75% 아래로 계속 하락하고 eval도 따라 내려가면 즉시 중지. 그 체크포인트로 resume하지 말 것.

## 6. 로그 인덱스

경로 prefix: `logs/<id>-isaaclab_pick_place_cube_plate_ppo_openpi_pi05_cri/`

| id | 한 줄 |
|---|---|
| `20260907-06:24:28` | CRI -1e-4, warmup 6, lr 1e-7. 91%→80% / 36 |
| `20260907-14:07:05` | CRI -1e-3, warmup 30, lr 5e-8. 91%→76% / 46. YAML 조정의 대표 실패 |
| `20260908-00:13:45` | 빈 런치 |
| `20260908-00:14:54` | noise 0.1 + GAE. KL/clip 폭주, 90%→88% / 10 |
| `20260908-02:21:10`, `04:53:17` | raw/actor, warmup freeze로 backward 크래시 |
| `20260908-05:57:42` | raw/actor 수정 후. critic 없이 91.5%→88.3% / 8 |
| `20260908-07:46:54` | 빈 런치 |
| `20260908-07:47:07` | **현재 성공 전용.** CRI 0, 스택식 PPO. 89.4%→86.0% / 9 (진행 중) |
