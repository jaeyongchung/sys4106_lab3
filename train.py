
import argparse
import numpy as np
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
from circuit import *
from simulator import *
from agent import *


# ──────────────────────────────────────────────────────────────
# Circuit graph을 그려주는 함수
# ──────────────────────────────────────────────────────────────
LEVEL_POS = {               # (x, y)  — y가 작을수록 아래쪽
    1: (0.0,  0),  2: (1.5,  0),      # level‑0
    4: (0.0, -1),  5: (1.5, -1),      # level‑1
    3: (0.75, -2),                    # level‑2
    0: (0.75, -3),                    # level‑3
}



def ascii_state_graph(ep, step, step_in_ep, s, done=False,
                      max_nets=6,
                      pins_per_net=2,
                      scale=4):
    if step_in_ep%CircuitEnv.MAX_STEPS in [0, 49]:
        print("#EPISODE = ", ep)
        print("#STEP IN EPISODE = ", step_in_ep)
        print("#STATE = ", s)
    
def visualize_state_graph(state_tuple,
                           ax,
                           max_nets=6,
                           pins_per_net=2):
    """flatten‑된 state(길이 6) → 고정 위치 그래프"""
    obs = np.asarray(state_tuple, int).reshape(max_nets, pins_per_net)

    G = nx.Graph()
    for pid in LEVEL_POS:               # 노드 0‑5 모두 생성
        G.add_node(pid)
    cmap = plt.get_cmap("Set2", max_nets)

    # 각 net(row)에 핀 2개가 모두 있으면 edge
    for nid in range(max_nets):
        pins = [p for p in obs[nid] if p >= 0]
        if len(pins) == 2:
            G.add_edge(pins[0], pins[1], color=cmap(nid))

    ax.clear()
    pos   = {n: LEVEL_POS[n] for n in G.nodes}
    edges = G.edges()
    ecols = [G[u][v]["color"] for u, v in edges] if edges else []

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#ADD8E6",
                           edgecolors="black", node_size=800)
    nx.draw_networkx_labels(G, pos, labels={n: f"P{n}" for n in G.nodes},
                            font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges,
                           edge_color=ecols, width=2, ax=ax)
    ax.set_title("Fixed‑level Netlist");  ax.axis("off")

# ──────────────────────────────────────────────────────────────
# Q‑table 히트맵을 그려주는 함수
# ──────────────────────────────────────────────────────────────
def draw_qtable(agent, ax, max_states=30):
    ax.clear()
    actions = sorted({a for qs in agent.Q.values() for a in qs})
    states  = list(agent.Q.keys())[:max_states]

    data = np.full((len(states), len(actions)), np.nan)
    for i, s in enumerate(states):
        for j, a in enumerate(actions):
            if a in agent.Q[s]:
                data[i, j] = agent.Q[s][a]

    vmax = np.nanmax(np.abs(data)) or 1
    im = ax.imshow(data, cmap="bwr", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(actions)))
    ax.set_xticklabels([str(a) for a in actions], rotation=90, fontsize=6)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels([str(s) for s in states], fontsize=6)
    ax.set_xlabel("Action")
    ax.set_ylabel("State")
    ax.set_title("Q‑table (first {} states)".format(len(states)))

    # 첫 호출 때 colorbar 추가
    if not hasattr(draw_qtable, "_cbar"):
        draw_qtable._cbar = plt.colorbar(im, ax=ax, pad=0.01)
    else:
        draw_qtable._cbar.update_normal(im)


# ──────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────

class QLearningGUI:
    def __init__(
        self,
        agent,
        viz_interval: int = 10,
        max_states_heat: int = 40
    ):
        self.agent            = agent
        self.viz_interval     = viz_interval
        self.max_states_heat  = max_states_heat
        self._build_fig()     # Figure / 위젯 생성
        self._running         = True
        self._last_step       = 0
        self._last_ep = 0

    # ---------- public 메소드 ----------
    def run_training(self, episodes=5_000):
        plt.ion();  plt.show()
        self.agent.train(episodes, callback=self._on_step)
        plt.ioff(); plt.show()   # STOP 후 마지막 화면 고정

    def run_testing(self):
        plt.ion();  plt.show()
        self.agent.test(callback=self._on_step_test)
        plt.ioff(); plt.show()   # STOP 후 마지막 화면 고정

    # ---------- 내부 (callback) ----------
    def _on_step(self, ep, step, step_in_ep, s, done):
        self._last_step = step
        self._last_ep = ep
        if ep % self.viz_interval == 0 and done:   # 에피소드 종료 시점
            self._draw(ep, s)
        return self._running      # False면 train() 에서 즉시 반환

    def _on_step_test(self, s):
        self._draw(self._last_ep, s)


    # ---------- Figure & Widgets ----------
    def _build_fig(self):
        self.fig  = plt.figure(figsize=(11, 6), layout="constrained")
        gs        = self.fig.add_gridspec(3, 2, height_ratios=[14, 1, 1])
        self.ax_g = self.fig.add_subplot(gs[0, 0])
        self.ax_q = self.fig.add_subplot(gs[0, 1])
        ax_sl     = self.fig.add_subplot(gs[1, :])
        ax_bt     = self.fig.add_subplot(gs[2, :])

        # ε 슬라이더
        self.slider = Slider(
            ax=ax_sl, label="ε  (exploration probability)",
            valmin=0.0, valmax=1.0,
            valinit=self.agent.eps, valstep=0.01
        )
        self.slider.on_changed(self._on_eps_change)

        # STOP 버튼
        self.btn = Button(ax_bt, label="STOP", color="#ee6666", hovercolor="#ffaaaa")
        self.btn.on_clicked(self._on_stop)

    def _on_eps_change(self, val):
        self.agent.eps = val

    def _on_stop(self, _event):
        self._running = False

    # ---------- 시각화 ----------
    def _draw(self, ep, s):
        visualize_state_graph(
            s, self.ax_g,
            self.agent.env.MAX_ARCS,
            self.agent.env.MAX_PINS_PER_NET
        )
        draw_qtable(self.agent, self.ax_q, self.max_states_heat)
        self.fig.suptitle(
            f"Episode {ep} | Step {self._last_step} | ε={self.agent.eps:.2f}",
            fontsize=12
        )
        plt.pause(0.001)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train QAgent on CircuitEnv with GUI slider.')
    
    # QAgent hyperparameters
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate for QAgent (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.98, help='Discount factor for QAgent (default: 0.98)')
    parser.add_argument('--eps', type=float, default=0.2, help='Epsilon for epsilon-greedy strategy (default: 0.2)')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes (default: 5000)')
    parser.add_argument('--gui', action='store_true', help='Enable GUI slider (default: False)')
    parser.add_argument('--final', action='store_true', help='Final Evaluation Mode (default: False)')

    args = parser.parse_args()


    if args.final:
        sum_re_pins = 0
        count = 0 
        for i in range(50):
            print("------- Trial %d ------------" % i)
            # write your target number
            env = CircuitEnv()
            agent = QAgent(env, alpha=args.alpha, gamma=args.gamma, eps=args.eps)
            agent.train(episodes=args.episodes) 
            reward, num_re_pins = agent.test()
            sum_re_pins += num_re_pins
            count += 1
        print("Your Score: %.3f" % (sum_re_pins/count))
    else:
        if not args.gui:
            agent.train(episodes=args.episodes, callback=ascii_state_graph)
            agent.test()
        else:
            gui = QLearningGUI(agent, viz_interval=20)
            gui.run_training(episodes=args.episodes)
            gui.run_testing()
