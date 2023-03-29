from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import numpy as np
from time import sleep
qmManager = QuantumMachinesManager()
import matplotlib.pyplot as plt
import numpy as np


f_res_guess1 = 6.8e9  # example resonator resonance frequency, Hz
f_res_guess2 = 6.95e9
f_res_guess3 = 6.78e9
f_res_guess4 = 6.94e9
f_res_guess5 = 6.85e9

f_res_guess = [f_res_guess1, f_res_guess2, f_res_guess3, f_res_guess4, f_res_guess5]

#starting qubit frequencies based on design parameters
f_qubit = [5.5e9, 5.56e9, 5.45e9, 5.89e9, 5.29e9]
f_IF_mixers = f_res_guess + f_qubit

# measurement parameters
# example LO frequency
f_LO = 7e9
Npts = 201
Amp = 0.2  # V

# IO parameters
opx_out_res_I = [1, 3, 5, 7, 9]
opx_out_res_Q = [2, 4, 6, 8, 10]
opx_in = 1
base_TOF = 180

'''
Example DC offsets for each IQ mixer. As of now, they are performed manually and manually inputted
into this file. In the future, this may be automated.
'''


DC_offset_I = [0.05, 0.01, 0.05, 0, 0.03]
DC_offset_Q = [0.05, 0.01, 0.05, 0, 0.03]

# demodulation parameters
demod_len = 1000
pulse_len = demod_len + 200  # nsec
pulse_ts = int(pulse_len / 4)
f0 = 10 / (demod_len * 1e-9)
print(f'f0={f0 * 1e-6}')
demod_ts = demod_len / 4
w = 8.0
w_full = [w] * int(demod_ts)
w_zero = [0.0] * int(demod_ts)

with program() as prog:
    I = declare(fixed)
    Q = declare(fixed)
    f = declare(int)

    with for_(f, int(f0), f <= int(2 * f0), f + int(0.001 * f0)):
        update_frequency('RR', f)
        measure('my_pulse', 'RR', 'samples', ('integ_w_cos', I), ('integ_w_sin', Q))
        save(I, 'I')
        save(Q, 'Q')
        save(f, 'f')

config = {
    'version': 1,
    'controllers': {
        'con1': {
            'type': 'opx1',
            'analog': {
                opx_out_res_I: {'offset': DC_offset_I},
                opx_out_res_Q: {'offset': DC_offset_Q},
            },
        },
    },

    'elements': {
        'RR1': {
            'mixInputs': {
                'I': ('con1', opx_out_res_I[0]),
                'Q': ('con1', opx_out_res_Q[0]),
                'lo_frequency': f_LO,
                'mixer': 'mixer1'
            },
            'frequency': f_res_guess[0],
            'operations': {
                'my_pulse': 'my_pulse_in',
                'play_pulse': 'play_pulse_in'
            },
            'time_of_flight': base_TOF + 12,
            'smearing': 0,
            'outputs': {
                'out1': ('con1', opx_in)
            }
        },
        'RR2': {
            'mixInputs': {
                'I': ('con1', opx_out_res_I[1]),
                'Q': ('con1', opx_out_res_Q[1]),
                'lo_frequency': f_LO,
                'mixer': 'mixer2'
            },
            'frequency': f_res_guess[1],
            'operations': {
                'my_pulse': 'my_pulse_in',
                'play_pulse': 'play_pulse_in'
            },
            'time_of_flight': base_TOF + 12,
            'smearing': 0,
            'outputs': {
                'out1': ('con1', opx_in)
            }
        },

        'RR3': {
            'mixInputs': {
                'I': ('con1', opx_out_res_I[2]),
                'Q': ('con1', opx_out_res_Q[2]),
                'lo_frequency': f_LO,
                'mixer': 'mixer3'
            },
            'frequency': f_res_guess[2],
            'operations': {
                'my_pulse': 'my_pulse_in',
                'play_pulse': 'play_pulse_in'
            },
            'time_of_flight': base_TOF + 12,
            'smearing': 0,
            'outputs': {
                'out1': ('con1', opx_in)
            }
        },

        'RR4': {
            'mixInputs': {
                'I': ('con1', opx_out_res_I[3]),
                'Q': ('con1', opx_out_res_Q[3]),
                'lo_frequency': f_LO,
                'mixer': 'mixer4'
            },
            'frequency': f_res_guess[3],
            'operations': {
                'my_pulse': 'my_pulse_in',
                'play_pulse': 'play_pulse_in'
            },
            'time_of_flight': base_TOF + 12,
            'smearing': 0,
            'outputs': {
                'out1': ('con1', opx_in)
            }

        },
        'RR5': {
            'mixInputs': {
                'I': ('con1', opx_out_res_I[4]),
                'Q': ('con1', opx_out_res_Q[4]),
                'lo_frequency': f_LO,
                'mixer': 'mixer5'
            },
            'frequency': f_res_guess1,
            'operations': {
                'my_pulse': 'my_pulse_in',
                'play_pulse': 'play_pulse_in'
            },
            'time_of_flight': base_TOF + 12,
            'smearing': 0,
            'outputs': {
                'out1': ('con1', opx_in)
            }

        },

        'pulses': {
        'my_pulse_in': {
            'operation': 'measurement',
            'length': pulse_len,
            'waveforms': {
                'I': 'exc_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'integ_w_cos': 'integ_w_cos',
                'integ_w_sin': 'integ_w_sin',
            },
            'digital_marker': 'marker1'
        },
        'play_pulse_in': {
            'operation': 'control',
            'length': pulse_len,
            'waveforms': {
                'I': 'exc_wf',
                'Q': 'zero_wf'
            },
        }
    },

    'waveforms': {
        'exc_wf': {
            'type': 'constant',
            'sample': Amp
        },
        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },
    },

    'digital_waveforms': {
        'marker1': {
            'samples': [(1, pulse_len), (0, 0)]
        }
    },

    'integration_weights': {
        'integ_w_cos': {
            'cosine': w_full,
            'sine': w_zero,
        },
        'integ_w_sin': {
            'cosine': w_zero,
            'sine': w_full,
        }
    },

    'mixers': {
        'mixer1': [
            {'freq': f_res_guess1, 'lo_freq': f_LO, 'correction': [1.0, 0.0, 0.0, 1.0]}
        ]

    }
    }
}



qm1 = qmManager.open_qm(config)

job = qm1.execute(prog, duration_limit=100, data_limit=800000)
print(job.id())
import time

time.sleep(0.2)

res = job.get_results()
I_r = np.array(res.variable_results.I.data) * 2 ** 12
Q_r = np.array(res.variable_results.Q.data) * 2 ** 12
f_r = np.array(res.variable_results.f.data, dtype='double')  # in Hz

print(res.errors)

# %% plot demodulation result and compare to calculation

if 'samples' in res.raw_results.__dict__:
    adc_raw_dat = np.array(res.raw_results.samples.input1_data) / 2 ** 12
    adc_ts = np.array(res.raw_results.samples.ts_in_ns)
    plt.figure()
    for i in range(3):
        time_strip = adc_ts[pulse_len * i:pulse_len * (i + 1)]
        adc_dat = adc_raw_dat[pulse_len * i:pulse_len * (i + 1)]
        if i == 0:
            pulse_amp = np.max(adc_dat) - np.min(adc_dat)

        demod_mask = np.zeros_like(time_strip, dtype='double')
        demod_mask[:demod_len] = np.max(adc_dat)
        plt.subplot(3, 1, i + 1)
        plt.plot(time_strip,
                 adc_dat)
        plt.plot(time_strip, demod_mask)
        plt.xlabel('sampling time [ns]')
        plt.ylabel('raw[ADC scale]')
        plt.title(f'analog in demod data {i + 1}')
    plt.show()
else:
    pulse_amp = 0.34 * Amp / 0.4

phi = 0.8
tvec = np.linspace(0, demod_len, int(demod_len))

mag_calc = np.zeros_like(f_r)
f0_int = 1e-9 * f0
for i, fq in enumerate(f_r):
    fq *= 1e-9
    pulse_calc = 0.5 * pulse_amp * np.sin(2 * np.pi * fq * tvec + phi)
    int_sin = np.trapz(w * pulse_calc * np.sin(2 * np.pi * fq * tvec), tvec)
    int_cos = np.trapz(w * pulse_calc * np.cos(2 * np.pi * fq * tvec), tvec)
    mag_calc[i] = np.sqrt(int_sin ** 2 + int_cos ** 2)

plt.figure()
mag_meas = np.sqrt(I_r ** 2 + Q_r ** 2)
plt.plot(1e-6 * f_r, mag_meas, label='mag measured')
plt.plot(1e-6 * f_r, mag_calc, label='mag calculated')
yl = plt.ylim()
plt.ylim((0, yl[1]))
plt.title('demodulation result')
plt.ylabel(r'mag = $\sqrt{I^2+Q^2}$')
plt.xlabel('f_q [MHz]')
plt.legend()
plt.grid()
plt.show()

# %% Qubit configuration with 3 fixed qubits

config2 = {
    'version': 1,
    'controllers': {
        'con1': {
            'type': 'opx1',
            'analog': {
                i + 1: {'offset': 0.0} for i in range(10)
            }
        }
    },

    'elements': {
        **{f'res{i}': {
            'mixInputs': {
                'I': ('con1', 7),
                'Q': ('con1', 8),
                'lo_frequency': 5e9,
                'mixer': 'mixer_res',
            },
            'intermediate_frequency': f_res_guess[i],
            'operations': {
                'meas_pulse': 'meas_pulse_in',
            },
            'time_of_flight': 240,  # nsec
            'smearing': 0,  # nsec
            'outputs': {
                'out1': ('con1', 1)
            }
        } for i in range(5)
        },
        **{f'qubit{i}': {
            'mixInputs': {
                'I': ('con1', 1 + 2 * i),
                'Q': ('con1', 2 + 2 * i),
                'lo_frequency': 6e9,
                'mixer': f'mixer{i + 3}',
            },
            'intermediate_frequency': f_qubit[i],

            'operations': {
                'exc_pulse': 'exc_pulse_in',
            }
        } for i in range(3)
        },
        'flux_line': {
            'singleInput': {
                'port': ('con1', 9),
            },
            'intermediate_frequency': 0.0,
            'operations': {
                'exc_pulse': 'exc_pulse_in',
            }
        }
    },

    'pulses': {
        'meas_pulse_in': {
            'operation': 'measurement',
            'length': 100,
            'waveforms': {
                'I': 'exc_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'integ_w_sine': 'integ_w_sine',
                'integ_w_cosine': 'integ_w_cosine',
            },
            'digital_marker': 'marker1'
        },
        'exc_pulse_in': {
            'operation': 'control',
            'length': pulse_len,
            'waveforms': {
                'I': 'exc_wf',
                'Q': 'zero_wf'
            }
        },
    },

    'waveforms': {
        'exc_wf': {
            'type': 'constant',
            'sample': 0.4
        },
        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },
        'DRAGcos': {
            # to be specified precisely as 'type': '...',
            'sample': 0.4
        },
        'DRAGsin': {
            # to be specified precisely as 'type': '...',
            'sample': 0.4
        },
    },

    'digital_waveforms': {
        'marker1': {
            'samples': [(1, pulse_len), (0, 0)]
        }
    },

    'integration_weights': {
        'integ_w_cosine': {
            'cosine': [1.0] * pulse_len,
            'sine': [0.0] * pulse_len,
        },
        'integ_w_sine': {
            'cosine': [0.0] * pulse_len,
            'sine': [1.0] * pulse_len,
        }
    },

    'mixers': {
        **{
            f'mixer{i}': [
                {'intermediate_frequency': f_IF_mixers[i + 5], 'lo_frequency': 5e9, 'correction': [1.0, 0.0, 0.0, 1.0]}
            ] for i in range(3)
        }
    },
    'mixer_res': [
        {'intermediate_frequency': f_IF_mixers[i], 'lo_frequency': 5e9, 'correction': [1.0, 0.0, 0.0, 1.0]} for i in range(5)
    ]
}

qm2 = qmManager.open_qm(config2)

job2 = qm2.execute(prog, duration_limit=100, data_limit=800000)
print(job2.id())
import time

time.sleep(0.2)

res2 = job2.get_results()
I_r = np.array(res2.variable_results.I.data) * 2 ** 12
Q_r = np.array(res2.variable_results.Q.data) * 2 ** 12
f_r = np.array(res2.variable_results.f.data, dtype='double')  # in Hz

print(res2.errors)

