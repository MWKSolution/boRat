from mwk_logger import MwkLogger
import numpy as np

__version__ = '1.0.0'

__log__ = MwkLogger(name='mech', stream_level='INFO').logger

np.set_printoptions(suppress=True, precision=6)

tolerance = 1e-12

""""
Euler angles rotations helper:
extrinsic = 'zyx' --->
intrinsic = 'XYZ' <---

right hand rotations helper:
  X ( Y -> Z)
  Y ( Z -> X)
  Z ( X -> Y)
"""



