import numpy as np
import logging

RADAR_COLOR = 'b'
RADAR_WITDH = 0.5
RADAR_MAX_LENGTH = 50
RADAR_NOISE_STD = 0.2

SYSTEM_MOTION_NOISE = [0.01, 0.01, 0.00]
SYSTEM_MEASURE_MODEL_LOCAL_NOISE_STD = 20
SYSTEM_MAP_OCCUPIED_AREA_THRESHOLD = 0.7 # 1 mean traversable
SYSTEM_NO_SENSORS = 7
SYSTEM_MC_INTEGRAL_GRID = 1
SYSTEM_MC_GRIDS = np.arange(0, RADAR_MAX_LENGTH + SYSTEM_MC_INTEGRAL_GRID, SYSTEM_MC_INTEGRAL_GRID)

ROBOT_MAX_MOVE_DISTANCE = 2.5
ROBOT_DEFAULT_X = 0
ROBOT_DEFAULT_Y = 0
ROBOT_DEFAULT_THETA = np.pi / 2.0
ROBOT_SYMBOL = 'bo'
ROBOT_APPROXIMATED_SYMBOL = 'ro'
ROBOT_APPROXIMATED_DIRECTION_LENGTH = 5
ROBOT_SYMBOL_SIZE = 6
ROBOT_STEP_SIZE = 0.3
ROBOT_STEP_THETA_SIZE = np.pi / 2.0
ROBOT_DIAMETER = 5
ROBOT_MOVEMENT_MODEL_PARAMS = [0.01, 0.01]


SENSOR_SYMBOL = 'gs'
SENSOR_SYMBOL_SIZE = 5

PARTICLE_SYMBOL = 'ro'
PARTICLE_SYMBOL_SIZE = 2
PARTICLE_DIRECTION_DISTANCE = 3
PARTICLE_OPACITY = 0.5

LOG_LEVEL = logging.INFO

SCENCES = {
    'scene-1': {
        'landmarks': [
            (165, 100-20, 2*np.pi-0.5*np.pi),
            (165, 100-80, np.pi), # next move theta
            (40, 100-80, 0.5*np.pi),
            (40, 100-70, 0.0),
            (110, 100-70, 0.5*np.pi),
            (110, 100-35, np.pi),
            (20, 100-35, np.pi),
        ]
    },
    'scene-2': {
        'landmarks': [
            (135, 200-90, np.pi),
            (80, 200-85, 3*np.pi/2),
            (80, 200-125, np.pi),
            (55, 200-125, np.pi/2),
            (55, 200-85, np.pi),
            (25, 200-85, 3*np.pi/2),
            (25, 200-170, np.pi/4),
            (50, 200-170, -np.pi/4),
            (80, 200-170, 0),
            (155, 200-170, np.pi/4),
            (195, 200-150, np.pi/2),
            (195, 200-105, np.pi/2),

        ]
    }
}
