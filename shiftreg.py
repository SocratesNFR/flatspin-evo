import numpy as np
from flatspin.model import SpinIce, CustomSpinIce
from flatspin.encoder import Encoder, scale, grid, fixed_vector

def spin2bin(spin):
    return (1 + spin) // 2

def bin2spin(bin):
    return bin * 2 - 1

class ShiftRegister(SpinIce):
    def __init__(self, *, size=4, sw_b=1, sw_c=1, sw_beta=3, sw_gamma=3, neighbor_distance=np.inf,
                 angle_b=-22, angle_d=22, angle_i=None,
                 orientation='horizontal', direction=1,
                 edge='symmetric', pitch_hor=1, pitch_ver=1, **kwargs):
        kwargs['size'] = size
        kwargs['sw_b'] = sw_b
        kwargs['sw_c'] = sw_c
        kwargs['sw_beta'] = sw_beta
        kwargs['sw_gamma'] = sw_gamma
        kwargs['neighbor_distance'] = neighbor_distance

        self.angle_b = angle_b # angle of buffer magnets
        self.angle_d = angle_d # angle of data magnets
        self.angle_i = angle_i if angle_i is not None else angle_d # angle of input magnet
        self.orientation = orientation
        self.direction = direction
        self.edge = edge
        self.pitch_hor = pitch_hor
        self.pitch_ver = pitch_ver
        assert orientation in ('horizontal', 'vertical')
        assert direction in (-1, 1)
        assert edge in ('symmetric', 'asymmetric')

        super().__init__(**kwargs)

    def _init_geometry(self):
        spin_count = 2 * self.size
        if self.edge == 'symmetric':
            spin_count += 1

        pos = np.zeros((spin_count, 2), dtype=float)
        angle = np.zeros(spin_count, dtype=float)

        if self.orientation == 'horizontal':
            pos[:,0] = self.direction * np.arange(spin_count)
            pos[:,0] *= self.pitch_hor * np.cos(np.deg2rad(self.angle_d))
            pos[1::2,1] = self.direction * self.pitch_hor * np.sin(np.deg2rad(self.angle_d))
            if self.direction == -1:
                pos[:,1] += self.pitch_hor * np.sin(np.deg2rad(self.angle_d))
        else:
            pos[:,1] = self.direction * np.arange(spin_count) * self.pitch_ver * np.cos(np.deg2rad(-self.angle_d))
            pos[1::2,0] = self.direction * self.pitch_ver * np.sin(np.deg2rad(-self.angle_d))
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

'''
def make_ringbuf(width=4, height=4, angle_d = 22.5, angle_b = -22.5, **params):
    kwargs = dict(
        angle_d = angle_d,
        angle_b = angle_b,
        edge = 'asymmetric',
        init = 1,
    )

    model_hlr = ShiftRegister(size=width, direction=1, **kwargs)
    model_hrl = ShiftRegister(size=width, direction=-1, **kwargs)
    model_vbt = ShiftRegister(size=height, orientation='vertical', direction= 1, pitch_ver=.8, **kwargs)
    model_vtb = ShiftRegister(size=height, orientation='vertical', direction=-1, pitch_ver=.8, **kwargs)

    pos_hlr = model_hlr.pos.copy()
    angle_hlr = model_hlr.angle.copy()
    pos_vbt = model_vbt.pos.copy()
    angle_vbt = model_vbt.angle.copy()
    pos_hrl = model_hrl.pos.copy()
    angle_hrl = model_hrl.angle.copy()
    pos_vtb = model_vtb.pos.copy()
    angle_vtb = model_vtb.angle.copy()

    pos = pos_hlr
    angle = angle_hlr

    pos_vbt[:,0] += np.max(pos_hlr[:,0]) - np.cos(np.deg2rad(90+model_hlr.angle_d))
    pos_vbt[:,1] += np.max(pos_hlr[:,1]) + 1
    pos = np.concatenate([pos, pos_vbt])
    angle = np.concatenate([angle, angle_vbt])

    pos_hrl[:,0] += np.max(pos_hlr[:,0]) - 1
    pos_hrl[:,1] += np.max(pos[:,1])
    pos = np.concatenate([pos, pos_hrl])
    angle = np.concatenate([angle, angle_hrl])

    pos_vtb[:,1] += np.max(np.abs(pos_vtb[:,1]))# - 1
    pos_vtb[:,1] += np.max(pos_hlr[:,1])
    pos_vtb[:,0] -= 1 - np.cos(np.deg2rad(90+model_hlr.angle_d))
    pos = np.concatenate([pos, pos_vtb])
    angle = np.concatenate([angle, angle_vtb])

    params.setdefault('sw_b', 1)
    params.setdefault('sw_c', 1)
    params.setdefault('sw_beta', 3)
    params.setdefault('sw_gamma', 3)

    model = CustomSpinIce(magnet_coords=pos, magnet_angles=angle, radians=True, **params)

    return model
'''

def make_ringbuf(width=4, height=4, angle_d = 22.5, angle_b = -22.5, pitch_hor=1, pitch_ver=1, **params):
    kwargs = dict(
        angle_d = angle_d,
        angle_b = angle_b,
        edge = 'asymmetric',
        init = 1,
    )

    model_hlr = ShiftRegister(size=width, direction=1, pitch_hor=pitch_hor, 
                              pitch_ver=pitch_ver, **kwargs)
    model_hrl = ShiftRegister(size=width, direction=-1, pitch_hor=pitch_hor,
                              pitch_ver=pitch_ver, **kwargs)
    model_vbt = ShiftRegister(size=height, orientation='vertical', direction=1,
                              pitch_hor=pitch_hor, pitch_ver=pitch_ver, **kwargs)
    model_vtb = ShiftRegister(size=height, orientation='vertical', direction=-1, 
                              pitch_hor=pitch_hor, pitch_ver=pitch_ver, **kwargs)
    
    pos_hlr = model_hlr.pos.copy()
    angle_hlr = model_hlr.angle.copy()
    pos_vbt = model_vbt.pos.copy()
    angle_vbt = model_vbt.angle.copy()
    pos_hrl = model_hrl.pos.copy()
    angle_hrl = model_hrl.angle.copy()
    pos_vtb = model_vtb.pos.copy()
    angle_vtb = model_vtb.angle.copy()

    pos = pos_hlr
    angle = angle_hlr

    pos_vbt[:,0] += np.max(pos_hlr[:,0]) + pitch_ver*np.sin(np.deg2rad(model_vbt.angle_d))
    pos_vbt[:,1] += np.max(pos_hlr[:,1]) + pitch_ver*np.cos(np.deg2rad(model_vbt.angle_d))
    pos = np.concatenate([pos, pos_vbt])
    angle = np.concatenate([angle, angle_vbt])

    pos_hrl[:,0] += np.max(pos_hlr[:,0]) - pitch_hor*np.cos(np.deg2rad(model_hrl.angle_d))
    pos_hrl[:,1] += np.max(pos[:,1])
    pos = np.concatenate([pos, pos_hrl])
    angle = np.concatenate([angle, angle_hrl])


    pos_vtb[:,0] += np.min(pos_hrl[:,0]) - pitch_ver * np.sin(np.deg2rad(model_vtb.angle_d))
    pos_vtb[:,1] += np.min(pos_hrl[:,1]) - pitch_ver * np.cos(np.deg2rad(model_vtb.angle_d))
    pos = np.concatenate([pos, pos_vtb])
    angle = np.concatenate([angle, angle_vtb])

    params.setdefault('sw_b', 1)
    params.setdefault('sw_c', 1)
    params.setdefault('sw_beta', 3)
    params.setdefault('sw_gamma', 3)

    model = CustomSpinIce(magnet_coords=pos, magnet_angles=angle, radians=True, **params)

    return model

'''
ringbuf = make_ringbuf(3, 3, alpha=0.001, neighbor_distance=10, pitch_ver=1.5, pitch_hor=2)
ringbuf.plot();
'''
