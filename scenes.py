import numpy as np
import config

class map_1_basic(object):
    def __init__(self):
        self.map = 'map_1.png'
        self.landmarks = [
            (165, 100-20, 2*np.pi-0.5*np.pi),
            (165, 100-74, np.pi), # next move theta
            (34, 100-74, 0.5*np.pi),
            (34, 100-63, 0.0),
            (108, 100-63, 0.5*np.pi),
            (108, 100-33, np.pi),
            (24, 100-33, np.pi),
        ]

        self.controls = map_1_basic._build_control(self.landmarks)
        self.total_frames = len(self.controls)

        self.no_sensors = 6

        print('we have %d controls' % len(self.controls))
        # print(self.controls)

        self.state_idx = 0

    def get_control(self):
        control = self.controls[self.state_idx]
        self.state_idx = self.state_idx + 1
        return control

    def move(self, robot_pos):
        control = self.get_control()

        nx = robot_pos[0] + np.where(robot_pos[2] > 0.5*np.pi and robot_pos[2] < 3.0/2.0*np.pi, -control[0], control[0])
        ny = robot_pos[1] + np.where(robot_pos[2] <= np.pi, control[1], -control[1])
        ntheta = (2*np.pi + robot_pos[2] + control[2]) % (2*np.pi)
        new_state = (nx, ny, ntheta)

        print("=======")
        print("control", control)
        print("new state", new_state)

        return new_state, control


    @staticmethod
    def _build_control(landmarks):

        controls = []
        for i in range(1, len(landmarks)):
            prev_lm = landmarks[i-1]
            curr_lm = landmarks[i]

            x_move = [config.ROBOT_MAX_MOVE_DISTANCE]*int(np.abs((curr_lm[0] - prev_lm[0]) / config.ROBOT_MAX_MOVE_DISTANCE))
            y_move = [config.ROBOT_MAX_MOVE_DISTANCE]*int(np.abs((curr_lm[1] - prev_lm[1]) / config.ROBOT_MAX_MOVE_DISTANCE))
            max_moves = np.max([len(x_move), len(y_move)])

            x_move = x_move + [0]*(max_moves - len(x_move))
            y_move = y_move + [0]*(max_moves - len(y_move))
            theta_move = [0]*(max_moves-1) + [curr_lm[2] - prev_lm[2]]

            cc = list(zip(x_move, y_move, theta_move))
            controls = controls + cc
        return controls
