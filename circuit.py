
# ---------------------------------
# Simple Digital Circuit Model
# ---------------------------------
class Pin:
    DRIVE = 0
    LOAD = 1
    GATE = 0
    PORT = 1

    id = 0

    def __init__(self, typ, superset):
        self.type = typ
        self.superset = superset
        self.nets = []
        self.in_arcs = []
        self.out_arcs = []
        self.id = Pin.id
        Pin.id += 1

    def __repr__(self):
        return str(self.id)

class Net:
    id = 0

    def __init__(self):
        self.drivers = []
        self.loads = []
        self.id = Net.id
        Net.id += 1
    
    def pins(self):
        return self.drivers + self.loads

    def is_dangling(self):
        return len(self.drivers)==0 or len(self.loads)==0

    def has_driver(self):
        return len(self.drivers)!=0

    def has_multiple_drivers(self):
        return len(self.drivers)>1

    def connect(self, pin):
        if pin.type==Pin.DRIVE:
            self.drivers.append(pin)
        else:
            self.loads.append(pin)
        pin.nets.append(self)
        assert len(pin.nets)<=1

    def __repr__(self):
        return str(self.id)

# Arcs are pin-to-pin connections
class Arc:
    id = 0

    def __init__(self, src, dest):
        self.src = src
        self.dest = dest
        self.src.out_arcs.append(self)
        self.dest.in_arcs.append(self)

        self.id = Arc.id
        Arc.id += 1
        
    def __repr__(self):
        return "(%d -> %d)" % (self.src.id, self.dest.id)

class Gate:

    id = 0

    def __init__(self, pins):
        self.pins = pins

        self.id = Gate.id
        Gate.id += 1
       
    def input_pins(self):
        return [pin for pin in self.pins if pin.type==Pin.LOAD]

    def output_pins(self):
        return [pin for pin in self.pins if pin.type==Pin.DRIVE]

    def __repr__(self):
        return str(self.id)
    

