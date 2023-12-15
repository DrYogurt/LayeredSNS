import torch
import torch.nn as nn
import copy

class GsynModule(nn.Module):
    def __init__(self, n, activation_fn, G_min=None, G_max=None, G_scale=None):
        super(GsynModule, self).__init__()
        
        n = (n, n) if type(n) is int else n
        
        # Default initialization
        if G_min is None:
            G_min = torch.rand(n, dtype=torch.double)
        if G_max is None:
            G_max = torch.rand(n, dtype=torch.double)
        if G_scale is None:
            G_scale = torch.rand(n, dtype=torch.double)
        
        self.G_min = nn.Parameter(G_min)
        self.G_max = nn.Parameter(G_max)
        self.G_scale = nn.Parameter(G_scale)
        
        assert G_min.size() == G_max.size() == G_scale.size()
        self.activation_fn = activation_fn

    def forward(self, u):
        row_repeat_u = u.unsqueeze(-1).expand(*u.size(), u.size(-1))
        unscaled_u = (row_repeat_u - self.G_min) / (self.G_max - self.G_min)
        activated_u = self.activation_fn(unscaled_u)
        scaled_u = activated_u * (self.G_max - self.G_min) + self.G_min
        return scaled_u * self.G_scale # sum over each column to get a single value for each row 

class ConductanceLayerMulti(nn.Module):
    def __init__(self, n, n_prev, dt=None,
                 C_mem=None, G_mem=None, b_mem=None, 
                 Esyn_self=None, Esyn_prev=None,
                 Gsyn_self=None, Gsyn_prev=None,
                 is_first=False
                ):
        super(ConductanceLayerMulti, self).__init__()
        
        # Default initialization
        if dt is None:
            dt = torch.ones(n, dtype=torch.double)
        if C_mem is None:
            C_mem = torch.rand(n, dtype=torch.double)
        if G_mem is None:
            G_mem = torch.rand(n, dtype=torch.double)
        if b_mem is None:
            b_mem = torch.rand(n, dtype=torch.double)
        if Esyn_self is None:
            Esyn_self = torch.rand(n, n, dtype=torch.double)
        if Esyn_prev is None:
            Esyn_prev = torch.rand(n, n, dtype=torch.double)
        
        self.n = n
        self.is_first = is_first
        self.dt = dt
        self.C_mem = nn.Parameter(C_mem)
        self.G_mem = nn.Parameter(G_mem)
        self.b_mem = nn.Parameter(b_mem)
        self.Esyn_self = nn.Parameter(Esyn_self)
        self.Esyn_prev = nn.Parameter(Esyn_prev)
        
        # Gsyn modules
        if Gsyn_self is None:
            self.Gsyn_self = GsynModule(n, activation_fn=lambda x: torch.clamp(x, min=0, max=1))
        else:
            self.Gsyn_self = Gsyn_self
        
        if Gsyn_prev is None:
            self.Gsyn_prev = GsynModule((n, n_prev), activation_fn=lambda x: torch.clamp(x, min=0, max=1))
        else:
            self.Gsyn_prev = Gsyn_prev

    def forward(self, u_self, u_prev):
        row_repeat_u_self = u_self.unsqueeze(-1).expand(*u_self.size(), u_self.size(-1))
        
        # here, we should be finding the input to each neuron i_syn = G_syn * (E_syn - u)
        ds = self.Esyn_self - row_repeat_u_self #TODO check this guy
        ds = self.Gsyn_self(u_self) * ds #TODO check this guy
        ds = torch.sum(ds, dim=1) # sum over each column to get a single value for each row

        if self.is_first:
            dp = u_prev #if the layer is first, dp is the input
        else:
            dp = self.Esyn_prev - row_repeat_u_self
            dp = self.Gsyn_prev(u_prev) * dp
            dp = torch.sum(dp, dim=1) # sum over each column to get a single value for each row

        assert ds.shape == dp.shape and ds.shape == u_self.shape
        
        du = (-self.G_mem * u_self + self.b_mem + ds + dp) * (self.dt / self.C_mem)
        return u_self + du

class ConductanceNetwork(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(ConductanceNetwork, self).__init__()
        self.layers = layers
        # previous states are 1 indexed, and state '0' is used to store the inputs
        self.current_states = [torch.zeros(layer.n) for layer in self.layers]
        self.prev_states = [state.clone().detach() for state in self.current_states]

    def forward(self, inp):
        self.prev_states.insert(0, inp)
        for i, layer in enumerate(self.layers):
            self.current_states[i] = layer(self.prev_states[i+1], self.prev_states[i])
        self.prev_states = [state.clone().detach() for state in self.current_states]
        assert(len(self.current_states) == len(self.layers))
        return self.current_states[-1]
