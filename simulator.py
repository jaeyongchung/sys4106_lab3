
import gym
import numpy as np
from gym import spaces
from collections import deque
from circuit import *


def action2str(a):
    if a[0]==0:
        return "create_gate"
    elif a[0]==1:
        return "connect pin %d to pin %d" % (a[1], a[2])

# ---------------------------------
# Simulator for RL
# ---------------------------------
class CircuitEnv(gym.Env):
    """OpenAI Gym‑style environment for a toy circuit construction task.

    Observation: a (MAX_ARCS x 2) matrix of pin indices per net (–1 if empty slot).
    Action space:
      0: create gate (adds 3 pins),
      1: connect pin to pin (2 arguments)
    """
    metadata = {"render.modes": ["human"]}

    # constants
    MAX_GATES = 1
    MAX_PINS = 6 # 0–2 reserved for ports, and 3 pins for the gate
    MAX_NETS = 3   
    MAX_ARCS = 6
    MAX_STEPS = 6
    MAX_PINS_PER_NET = 2 # We will use two-pin nets only

    PIN_OUTPUT_PORT = 0
    PIN_INPUT1_PORT = 1
    PIN_INPUT2_PORT = 2

    def __init__(self):
        super().__init__()
        ## YOU MAY WANT TO CHANGE THIS ACTION SPACE

        self.action_space = spaces.MultiDiscrete([2, self.MAX_PINS , self.MAX_PINS]) 
        
        # A directed graph is often represented by a list of directed edges, 
        # and an directed edge is represented by a tuple of the src pin and the destination pin.
        # the following observation space is based on this representation for the circuit graph.

        ## YOU ARE FREE TO CHANGE OBSERVATION SPACE, BUT IT WOULD BE DIFFICULT
        self.observation_space = spaces.Box(
            low=-1, # -1 means "not exist"
            high=self.MAX_PINS - 1, # 0~5
            shape=(self.MAX_ARCS, 2), 
            dtype=np.int32,
        )
        self.reset()

    def reset(self, *, seed=9, options=None):
        super().reset(seed=seed)
        # initial ports
        Pin.id = 0
        Gate.id = 0
        Net.id = 0
        Arc.id = 0
        self.pins = {
            0: Pin(Pin.LOAD, Pin.PORT) ,
            1: Pin(Pin.DRIVE, Pin.PORT),
            2: Pin(Pin.DRIVE, Pin.PORT)
        }
        self.gates = {}
        self.nets = {}
        self.arcs = {} 
        self._steps = 0
        self._num_reachable = len(self._reachable_pins()) # for output port
        self.render()
        return self._get_obs(), {}


    # ──────────────────────────────────────────────────────────────
    # 1) 0번 핀에서 닿을 수 있는 모든 핀 집합 구하기
    def _reachable_pins(self):
        """BFS ‑ pin 0 에서 연결(넷)로 따라가며 reachable 핀 집합 반환"""

        q   = deque([self.pins[self.PIN_OUTPUT_PORT]])  # 즉 0번
        vis = set()
        while q:
            p = q.popleft()
            if p in vis:
                continue
            vis.add(p)
            for arc in p.in_arcs:
                other = arc.src
                if other not in vis:
                    q.append(other)
        return vis                    # {0, …}

    def reward_function(self, action):
        """
        보상 = pin 0 으로부터 연결된 핀 총 개수 – 자기 자신 포함
               (예: 0번만 연결 → 1점, 0·1·2 연결 → 3점)
        """
        a_type, pin_arg, net_arg_p1 = action
        ## TODO: WRITE YOUR REWARD FUNCTION HERE
        return 0

    

    def step(self, action):
        a_type, pin_arg, net_arg_p1 = action

        ## YOU MAY WANT TO CHANGE THE FOLLOWING LINES FOR MORE COMPLICATED REWARD FUNCTION
        self._steps += 1
        reward = 0
        if a_type == 0:
            score = self._create_gate()
        elif a_type == 1:
            score = self._create_net()
            if score>=0:
                nid = score
                ret = self._connect(pin_arg, nid) 
                if ret>=0:
                   self._connect(net_arg_p1, nid)
                    
        reward = self.reward_function(action)
        done =  False
            
        if self._steps >= self.MAX_STEPS:
            done = True

        return self._get_obs(), reward, done, False, {}

    def _create_gate(self):
        if len(self.gates) >= self.MAX_GATES:
            return -1
        if len(self.pins) + 3 > self.MAX_PINS: # 3 is the number of the ports
            return -1
        a = Pin(Pin.DRIVE, Pin.GATE)
        b = Pin(Pin.LOAD, Pin.GATE)
        c = Pin(Pin.LOAD, Pin.GATE) 
        self.pins[a.id] = a
        self.pins[b.id] = b
        self.pins[c.id] = c     

        g = Gate([a,b,c])
        self.gates[g.id] = g
        # create the two internal pin-to-pin connections of the gate
        arc1 = self._create_arc(b, a) 
        assert arc1>=0
        arc2 = self._create_arc(c, a) 
        assert arc2>=0
        return g.id

    def _create_arc(self, src, dest):
        if len(self.arcs) >= self.MAX_ARCS:
            return -1
        
        a = Arc(src, dest)
        self.arcs[a.id] = a
        return a.id

    def _create_net(self):
        if len(self.nets) >= self.MAX_NETS:
            return -1
        net_id = len(self.nets)
        n = Net()
        self.nets[n.id] = n
        return n.id

    def _connect(self, pin_idx: int, net_idx: int):
        if pin_idx not in self.pins or net_idx not in self.nets:
            return -1

        net = self.nets[net_idx]
        pin = self.pins[pin_idx]
        if len(pin.nets)!=0: 
            #print("Warning: the pin '%d' is already connected to a net." % pin.id)
            return -1

        if net.has_driver() and pin.type==Pin.DRIVE:
            #print("Warning: the net '%d' has multiple drivers." % net.id)
            return -1

        net.connect(pin)
        if pin.type == Pin.DRIVE:
            #print(net.loads)
            for load in net.loads:
                self._create_arc(pin, load)
        else:
            #print(net.drivers)
            for driver in net.drivers:
                self._create_arc(driver, pin)

        return 0
        
        
    def _get_obs(self):
        # build MAX_NETS x 2 array, fill with -1 if fewer pins
        obs = np.full((self.MAX_ARCS, 2), -1, dtype=np.int32)
        for i, arc in self.arcs.items():
            obs[i] = (arc.src.id, arc.dest.id)
        return obs

    def render(self):
        pass


