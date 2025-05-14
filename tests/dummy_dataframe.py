'''Dummy DataFrame to test features with a mix of types and or
negative, zero, positive, nan, inf and large values.'''

import numpy as np
import pandas as pd

largef=1.7976931348623157e+300
smallf=2.2250738585072014e-300

DUMMY_DF = pd.DataFrame({
    'strings':                   ['a',     'b',     'c',     'd',     'e',     'f',     'd'    ],
    'ints_pos':                  [ 1,       2,       3,       4,       5,       6,       7     ],
    'ints_pos_zero':             [ 0,       1,       3,       4,       5,       6,       7     ],
    'ints_neg':                  [-7,      -6,      -5,      -4,      -3,      -2,      -1     ],
    'ints_neg_zero':             [-6,      -5,      -4,      -3,      -2,      -1,       0     ],
    'ints_neg_pos_zero':         [-3,      -2,      -1,       0,       1,       2,       3     ],
    'floats_pos':                [ 1.0,     2.0,     3.0,     4.0,     5.0,     6.0,     7.0   ],
    'floats_pos_zero':           [ 0.0,     1.0,     2.0,     3.0,     4.0,     5.0,     6.0   ],
    'floats_neg':                [-7.0,    -6.0,    -5.0,    -4.0,    -3.0,    -2.0,    -1.0   ],
    'floats_neg_zero':           [-6.0,    -5.0,    -4.0,    -3.0,    -2.0,    -1.0,     0.0   ],
    'floats_neg_pos_zero':       [-3.0,    -2.0,    -1.0,     0.0,     1.0,     2.0,     3.0   ],
    'ints_pos_nan':              [ 1,       2,       3,       4,       5,       6,       np.nan],
    'ints_pos_zero_nan':         [ 0,       1,       3,       4,       5,       np.nan,  7     ],
    'ints_neg_nan':              [-7,      -6,      -5,      -4,       np.nan, -2,      -1     ],
    'ints_neg_zero_nan':         [-6,      -5,      -4,      -3,      -2,      -1,       0     ],
    'ints_neg_pos_zero_nan':     [-3,      -2,      -1,       np.nan,  1,       2,       3     ],
    'floats_pos_nan':            [ 1.0,     2.0,     np.nan,  4.0,     5.0,     6.0,     7.0   ],
    'floats_pos_zero_nan':       [ 0.0,     1.0,     2.0,     3.0,     4.0,     5.0,     6.0   ],
    'floats_neg_nan':            [-7.0,     np.nan, -5.0,    -4.0,    -3.0,    -2.0,    -1.0   ],
    'floats_neg_zero_nan':       [ np.nan, -5.0,    -4.0,    -3.0,    -2.0,    -1.0,     0.0   ],
    'floats_neg_pos_zero_nan':   [-3.0,     np.nan, -1.0,     0.0,     1.0,     2.0,     3.0   ],
    'ints_pos_inf':              [ 1,       2,       3,       4,       5,       6,       np.inf],
    'ints_pos_zero_inf':         [ 0,       1,       3,       4,       5,       np.inf,  7     ],
    'ints_neg_inf':              [-7,      -6,      -5,      -4,       np.inf, -2,      -1     ],
    'ints_neg_zero_inf':         [-6,      -5,      -4,      -3,      -2,      -1,       0     ],
    'ints_neg_pos_zero_inf':     [-3,      -2,      -1,       np.inf,  1,       2,       3     ],
    'floats_pos_inf':            [ 1.0,     2.0,     np.inf,  4.0,     5.0,     6.0,     7.0   ],
    'floats_pos_zero_inf':       [ 0.0,     1.0,     2.0,     3.0,     4.0,     5.0,     6.0   ],
    'floats_neg_inf':            [-7.0,     np.inf, -5.0,    -4.0,    -3.0,    -2.0,    -1.0   ],
    'floats_neg_zero_inf':       [ np.inf, -5.0,    -4.0,    -3.0,    -2.0,    -1.0,     0.0   ],
    'floats_neg_pos_zero_inf':   [-3.0,     np.inf, -1.0,     0.0,     1.0,     2.0,     3.0   ],
    'floats_pos_large':          [ 1.0,     2.0,     largef,  4.0,     5.0,     6.0,     7.0   ],
    'floats_pos_zero_large':     [ 0.0,     1.0,     2.0,     3.0,     4.0,     5.0,     6.0   ],
    'floats_neg_large':          [-7.0,     largef, -5.0,    -4.0,    -3.0,    -2.0,    -1.0   ],
    'floats_neg_zero_large':     [ largef, -5.0,    -4.0,    -3.0,    -2.0,    -1.0,     0.0   ],
    'floats_neg_pos_zero_large': [-3.0,     largef, -1.0,     0.0,     1.0,     2.0,     3.0   ],
    'floats_pos_small':          [ 1.0,     2.0,     smallf,  4.0,     5.0,     6.0,     7.0   ],
    'floats_pos_zero_small':     [ 0.0,     1.0,     2.0,     3.0,     4.0,     5.0,     6.0   ],
    'floats_neg_small':          [-7.0,     smallf, -5.0,    -4.0,    -3.0,    -2.0,    -1.0   ],
    'floats_neg_zero_small':     [ smallf, -5.0,    -4.0,    -3.0,    -2.0,    -1.0,     0.0   ],
    'floats_neg_pos_zero_small': [-3.0,     smallf, -1.0,     0.0,     1.0,     2.0,     3.0   ],
    'constant':                  [ 1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0   ]
})
