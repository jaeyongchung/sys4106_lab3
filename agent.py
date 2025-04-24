
import numpy as np



def encode_state(obs: np.ndarray) -> tuple:
    # flatten matrix to a single tuple of ints
    return tuple(obs.flatten().tolist())


class QAgent:
    """
    간단한 ε-greedy Q-learning 에이전트
    -----------------------------------
    • train() : 배치/인터랙티브 학습
    • test()  : greedy 정책으로 rollout
    둘 다 encode_state()·callback 을 의존하므로
    외부 유틸은 그대로 두고, agent 만 교체하면 호환됩니다.
    """
    # ─────────────────── 초기화 ───────────────────
    def __init__(self, env, alpha=0.1, gamma=0.98, eps=0.2):
        self.env   = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps   = eps
        self.Q     = {}          # dict[state][action] = value
        self._running = False

    # ─────────────────── 내부 메서드 ───────────────────
    def _select_action(self, state):
        """ε-greedy 선택 (state 는 encode_state() 결과)"""
        if (
            np.random.rand() < self.eps or
            state not in self.Q or
            not self.Q[state]
        ):
            return tuple(self.env.action_space.sample())
        return max(self.Q[state], key=self.Q[state].get)

    # ─────────────────── 학습 (기존 train) ───────────────────
    # THIS IS SIMPLE EPSILON-GREEDY Q-LEARNING
    # YOU MAY WANT TO IMPROVE THIS
    def train(self, episodes=5_000, callback=None):
        """
        ε-greedy Q-learning 학습 루프
        --------------------------------
        parameters
        ----------
        episodes : int
            학습할 총 에피소드 수
        callback : Optional[Callable]
            callback(ep, step, s, done) → bool|None
            • 반환이 False 면 즉시 학습 중단
        """
        step = 0
        env  = self.env

        for ep in range(episodes):
            obs, _ = env.reset()
            s      = encode_state(obs)
            done   = False

            step_in_ep = 0
            while not done:
                a               = self._select_action(s)
                obs2, r, done, _, _ = env.step(a)
                s2              = encode_state(obs2)

                # ----- Q 업데이트 -----
                self.Q.setdefault(s,  {}).setdefault(tuple(a), 0.0)
                self.Q.setdefault(s2, {})
                nxt = max(self.Q[s2].values(), default=0.0)
                self.Q[s][tuple(a)] += self.alpha * (r + self.gamma * nxt
                                                     - self.Q[s][tuple(a)])

                # ----- 콜백 -----
                if callback and callback(ep=ep, step=step, step_in_ep=step_in_ep, s=s2, done=done) is False:
                    return    # STOP 신호
                s    = s2
                step_in_ep += 1
                step += 1

    # ─────────────────── 평가 / 데모 (기존 test) ───────────────────
    def test(self, max_steps=1_000, callback=None, greedy=True):
        """
        학습된 정책으로 rollout
        ----------------------
        greedy=True → ε=0 으로 두고 deterministic 실행
        callback(s) 형태로 상태별 훅 제공
        """
        env  = self.env
        obs, _ = env.reset()
        s      = encode_state(obs)
        done   = False
        total_r = 0.0
        
        def action2str(a):
            if a[0]==0:
                return "create_gate"
            elif a[0]==1:
                return "connect pin %d to pin %d" % (a[1], a[2])

        # ε 일시 변경 (greedy 모드)
        backup_eps = self.eps
        if greedy:
            self.eps = 0.0

        if callback:
            callback(s)

        print("Actions based on final policy from the initial state")
        for _ in range(max_steps):
            a       = self._select_action(s)
            print("Action: %s <%s>" % (action2str(a),str(a)))
            obs, r, done, _, _ = env.step(a)
            s       = encode_state(obs)
            if callback:
                callback(s)
            total_r += r
            if done:
                break

        num_re_pins = len(env._reachable_pins())
        # ε 복구
        self.eps = backup_eps
        print("Total reward:", total_r)
        print("Reachable Pins:", num_re_pins)
        return total_r, num_re_pins


