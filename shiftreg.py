import numpy as np
from flatspin.model import SpinIce
from flatspin.encoder import Encoder, scale, grid, fixed_vector

def spin2bin(spin):
    return (1 + spin) // 2

def bin2spin(bin):
    return bin * 2 - 1

class ShiftRegister(SpinIce):
    def __init__(self, *, sw_b=1, sw_c=1, sw_beta=3, sw_gamma=3, neighbor_distance=np.inf,
                 angle_b=-22, angle_d=22, angle_i=None, **kwargs):
        kwargs['sw_b'] = sw_b
        kwargs['sw_c'] = sw_c
        kwargs['sw_beta'] = sw_beta
        kwargs['sw_gamma'] = sw_gamma
        kwargs['neighbor_distance'] = neighbor_distance

        self.angle_b = angle_b # angle of buffer magnets
        self.angle_d = angle_d # angle of data magnets
        self.angle_i = angle_i if angle_i is not None else angle_d # angle of input magnet

        super().__init__(**kwargs)

    def _init_geometry(self):
        spin_count = 2 * self.size + 1

        pos = np.zeros((spin_count, 2), dtype=float)
        angle = np.zeros(spin_count, dtype=float)

        pos[:,0] = np.arange(spin_count)
        pos[1::2,1] = 0.5 * np.sin(np.deg2rad(self.angle_d))
        pos *= self.lattice_spacing

        angle[0::2] = np.deg2rad(self.angle_d)
        angle[1::2] = np.deg2rad(self.angle_b)
        angle[0] = np.deg2rad(self.angle_i)

        return pos, angle

    def _init_spin(self, init):
        self.spin *= -1
        super()._init_spin(init)

    def set_input(self, input):
        self.spin[0] = bin2spin(input)

    def get_input(self):
        return spin2bin(self.spin[0])

    def set_buf(self, buf):
        self.spin[1::2] = bin2spin(np.array(buf))

    def get_buf(self):
        return spin2bin(self.spin[1::2])

    def set_data(self, data):
        self.spin[2::2] = bin2spin(np.array(data))

    def get_data(self):
        return spin2bin(self.spin[2::2])


def shift_cycle(input, Hb=0.5, Hd=0.5, Hb_angle=22, Hd_angle=-22):
    Hb = Hb * np.array([np.cos(np.deg2rad(Hb_angle)), np.sin(np.deg2rad(Hb_angle))])
    Hd = Hd * np.array([np.cos(np.deg2rad(Hd_angle)), np.sin(np.deg2rad(Hd_angle))])

    out = np.repeat(input, 5, axis=0)
    out[1::5] = Hb
    out[2::5] = -Hb
    out[3::5] = Hd
    out[4::5] = -Hd

    return out

class ShiftRegEncoder(Encoder):
    """ Shift input into shift register

    Encodes input in three phases:
    1. Input is applied as local field over input magnet
    2. Global Hb field cycle
    3. Global Hd field cycle

    grid specifies where input field is to be applied
    H0 specifies the strength of the input field for 0
    H specifies the strength of the input field for 1 
    phi is the angle of the input field

    Hb and Hd is the strength of the Hb and Hd fields, respectively
    Hb_angle and Hd_angle specifies angle of the Hb and Hd fields, respectively
    """
    def __init__(self, H0=-1, H=1, phi=22, **params):
        params['H0'] = H0
        params['H'] = H
        params['phi'] = phi
        super().__init__(**params)

    steps = (scale, grid, fixed_vector, shift_cycle,)

class GlobalShiftRegEncoder(Encoder):
    """ Shift input into shift register

    Encodes input in three phases:
    1. Input is applied as global field
    2. Global Hb field cycle
    3. Global Hd field cycle

    H0 specifies the strength of the input field for 0
    H specifies the strength of the input field for 1 
    phi is the angle of the input field

    Hb and Hd is the strength of the Hb and Hd fields, respectively
    Hb_angle and Hd_angle specifies angle of the Hb and Hd fields, respectively
    """
    def __init__(self, H0=-1, H=1, phi=22, **params):
        params['H0'] = H0
        params['H'] = H
        params['phi'] = phi
        super().__init__(**params)

    steps = (scale, fixed_vector, shift_cycle,)
