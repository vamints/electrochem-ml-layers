"""
====================================================
Description : Butler Volmer Layer in Pytorch for Electrochemistry RNNs
Author      : Vladislav Mints
Created     : 2026-05-23
Version     : 1.0.0

BV_layer as described in https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-kxg13/v2

It uses the equations:

j_ox = e ^ ( j0 + a * (E - E_eq) )
j_ox = j_ox * active sites * gate
j_ox = j_ox * j_ox_lim / (|j_ox| + j_ox_lim)

j_red = e ^ ( j0 - a * (E - E_eq) )
j_red = j_red * active sites * gate
j_red = j_red * j_red_lim / (|j_red| + j_red_lim)

j = j_ox - j_red

INIT:

The BV layer takes as input for initialization:

active_site_dim = the number of dimensions that the active site distribution is modelled with
j0_lim (tuple) = the min and max for log-exchange current density (ln j₀). If known in linear scale, convert via ln(j₀) before passing.
tafel_slope_lim (tuple) = the min and max tafel slopes in mV/dec. The script will convert these to the right a values. 
E_eq_lim (tuple) = the min and max values for equilibrium potentials
j_lim_lim (tuple) = the maximum reductive limiting current and the maximum oxidative limiting current

red_gate_lim (tuple) = the min and max value for the potentials of the redox gate. This redox gate shuts down all reductive activity below the identified potential
ox_gate_lim (tuple) = the min and max value for the potentials of the oxide gate. This oxide gate shuts down all oxidative activity above the identified potential
when ox_gate_lim or red_gate_lim are (None,None) they will be ignored

gate_strength (tuple) = the sharpness of the reductive and oxide gates, respectively. i.e. how fast they shut down activity above a certain potential


FORWARD:

The forward function takes the inputs:
    E_next: (batch, seq_len, 1) — Electrode potential to be applied at the next timestep (t + 1)
    active_sites: (batch, seq_len, active_site_dim) — Active site distribution

    
NOTES:
    The exponential terms j_ox, j_red may overflow for extreme inputs or poorly initialized models.
    If this occurs, consider applying soft-bounding to the exponential argument.

====================================================
"""

import torch.nn as nn
import torch
import math

class BV_Layer(nn.Module):

    def __init__(self, active_site_dim, 
                 j0_lim=(-20,0), 
                 tafel_slope_lim = (30, 200), 
                 E_eq_lim = (0.3, 1.0),
                 j_lim_lim = (2, 10), 
                 red_gate_lim = (None, None), 
                 ox_gate_lim = (None, None),
                 gate_strength = (200,200)):

        super().__init__()

        #initialize active site distribution
        self.active_site_dim = active_site_dim

        #initialize starting values
        self.j0_raw = nn.Parameter(torch.randn(active_site_dim)*0.7)
        self.a_raw = nn.Parameter(torch.randn(active_site_dim))
        self.E_eq_raw = nn.Parameter(torch.randn(active_site_dim)*0.7)

        self.j0_min, self.j0_max = j0_lim
        tafel_slope_min, tafel_slope_max = tafel_slope_lim

        self.a_min = math.log(10) / (tafel_slope_max * 1e-3)
        self.a_max = math.log(10) / (tafel_slope_min * 1e-3)

        self.E_eq_min, self.E_eq_max = E_eq_lim
        
        self.j_red_max, self.j_ox_max = j_lim_lim

        self.red_gate_min, self.red_gate_max = red_gate_lim
        self.ox_gate_min, self.ox_gate_max = ox_gate_lim

        #initialize the sigmoid gates if they exist
        if all(v is not None for v in red_gate_lim):
            self.red_gate_raw = nn.Parameter(torch.randn(active_site_dim)*0.5)
        
        if all(v is not None for v in ox_gate_lim):
            self.ox_gate_raw = nn.Parameter(torch.randn(active_site_dim)*0.5)

        self.red_gate_strength, self.ox_gate_strength = gate_strength

    def forward(self, E_next, active_sites):

        #Expand the E next to match the dimensions
        E_next_exp = E_next.expand(-1, -1, self.active_site_dim)

        #perform the tanh transformation to get the values in the designated range
        j0 = 0.5 * (self.j0_max + self.j0_min) + 0.5 * (self.j0_max - self.j0_min) * torch.tanh(self.j0_raw)
        j0 = j0.view(1, 1, self.active_site_dim)

        a = 0.5 * (self.a_max + self.a_min) + 0.5 * (self.a_max - self.a_min) * torch.tanh(self.a_raw)
        a = a.view(1, 1, self.active_site_dim)

        E_eq = 0.5 * (self.E_eq_max + self.E_eq_min) + 0.5 * (self.E_eq_max - self.E_eq_min) * torch.tanh(self.E_eq_raw)
        E_eq = E_eq.view(1, 1, self.active_site_dim)

        #calculate the gates
        red_gate = torch.ones_like(E_next_exp)
        if hasattr(self, "red_gate_raw"):
            red_gate = 0.5 * (self.red_gate_max + self.red_gate_min) + 0.5 * (self.red_gate_max - self.red_gate_min) * torch.tanh(self.red_gate_raw)
            red_gate = red_gate.view(1, 1, self.active_site_dim)

            red_gate = torch.sigmoid(self.red_gate_strength * (E_next_exp-red_gate))
        
        ox_gate = torch.ones_like(E_next_exp)
        if hasattr(self, "ox_gate_raw"):
            ox_gate = 0.5 * (self.ox_gate_max + self.ox_gate_min) + 0.5 * (self.ox_gate_max - self.ox_gate_min) * torch.tanh(self.ox_gate_raw)
            ox_gate = ox_gate.view(1, 1, self.active_site_dim)

            ox_gate = torch.sigmoid(self.ox_gate_strength * (ox_gate - E_next_exp))

        #calculate the oxidative current
        j_ox = torch.exp(j0 + a * (E_next_exp - E_eq))
        j_ox = j_ox * active_sites * ox_gate

        #perform the Koutecky-Levich capping
        j_ox = (j_ox * self.j_ox_max) / (j_ox.abs() + self.j_ox_max)

        #calculate the reductive current
        j_red = torch.exp(j0 - a * (E_next_exp - E_eq))
        j_red = j_red * active_sites * red_gate

        #perform the Koutecky-Levich capping
        j_red = (j_red * self.j_red_max) / (j_red.abs() + self.j_red_max)

        #calculate total j
        j_total =  j_ox - j_red

        return j_total