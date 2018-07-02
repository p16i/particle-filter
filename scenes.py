import numpy as np
import config
import matplotlib.image as mpimg
from scipy import signal


import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=config.LOG_LEVEL)

class map_1_basic(object):
    def __init__(self, no_particles=20):

        self.no_particles = no_particles

        map_name = 'map_1.png'
        self.map = mpimg.imread(map_name)[::-1, :, 0]

        mark = np.ones((config.ROBOT_DIAMETER, config.ROBOT_DIAMETER))

        self.convolve_mark = (signal.convolve2d(1-self.map[:, :], mark, mode='same', boundary='fill', fillvalue=0)) / np.sum(mark)

        self.convolve_mark_overlay = np.copy(self.map)

        threshold = 1/np.sum(mark)
        self.convolve_mark_overlay[self.convolve_mark > threshold] = 0.2

        self.map_with_safe_boundary = np.copy(self.map)
        self.map_with_safe_boundary[self.convolve_mark > threshold] = 0.0 # zero is obstacle.

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

        self.no_sensors = 11
        self.radar_thetas = (np.arange(0, self.no_sensors) - self.no_sensors // 2)*(np.pi/self.no_sensors)

        logging.info('we have %d controls' % len(self.controls))
        # print(self.controls)

        self.traversable_area = np.stack(np.nonzero(1 - (self.map_with_safe_boundary.T < 0.7)), axis=1)
        particles_xy_indices = np.random.choice(self.traversable_area.shape[0], size=self.no_particles, replace=True)
        particles_xy = self.traversable_area[particles_xy_indices]

        particles_theta = np.random.uniform(0.0, 2*np.pi, (self.no_particles, 1))

        self.particles = np.hstack([particles_xy, particles_theta])

        self.state_idx = 0

    def get_control(self):
        control = self.controls[self.state_idx]
        self.state_idx = self.state_idx + 1
        return control

    def perform_control(self, pos, control):
        nx = pos[0] + np.where(pos[2] > 0.5 * np.pi and pos[2] < 3.0 / 2.0 * np.pi, -control[0], control[0])
        ny = pos[1] + np.where(pos[2] <= np.pi, control[1], -control[1])
        ntheta = (2 * np.pi + pos[2] + control[2]) % (2 * np.pi)

        nx = nx + np.random.normal(0, config.SYSTEM_MOVEMENT_NOISE[0])
        ny = ny + np.random.normal(0, config.SYSTEM_MOVEMENT_NOISE[1])
        ntheta = ntheta + np.random.normal(0, config.SYSTEM_MOVEMENT_NOISE[2])

        new_state = (
            nx,
            ny,
            ntheta
        )

        v = (nx - pos[0], ny - pos[1])

        return new_state, v

    def vperform_control(self, vpos, control):
        new_state, new_v = np.zeros(vpos.shape), np.zeros((vpos.shape[0], 2))

        for i in range(vpos.shape[0]):
            new_state[i], new_v[i] = self.perform_control(vpos[i], control)

        return new_state, new_v


    def raytracing(self, src, dest, num_samples=100, threshold=0.7):
        logging.debug('src %s -> dest %s ' % (','.join(src.astype(str)), ','.join(dest.astype(str))))

        dx = np.where(src[0] < dest[0], 1, -1)
        dy = np.where(src[1] < dest[1], 1, -1)
        x_steps = src[0] + dx*np.linspace(0, np.abs(src[0]-dest[0]), num_samples)
        y_steps = src[1] + dy*np.linspace(0, np.abs(src[1]-dest[1]), num_samples)

        x_steps_int = np.clip(np.round(x_steps).astype(np.int16), 0, self.map.shape[1]-1)
        y_steps_int = np.clip(np.round(y_steps).astype(np.int16), 0, self.map.shape[0]-1)

        mark = np.zeros(self.map.shape)
        mark[y_steps_int, x_steps_int] = 1

        collided_map = self.map[y_steps_int, x_steps_int] < threshold

        if np.sum(collided_map) > 0:
            collisions = np.nonzero(collided_map)
            pos = collisions[0][0]
            position = np.array((x_steps[pos], y_steps[pos]))
            logging.debug('    collided pos %s' % ','.join(position.astype(str)))
            distance = np.linalg.norm([position[0] - src[0], position[1] - src[1]])
        else:
            position = dest
            distance = config.RADAR_MAX_LENGTH

        rel_position = [
            (position[0] - src[0]),
            (position[1] - src[1]),
        ]
        logging.debug('  position %s' % ','.join(np.array(position).astype(str)))
        logging.debug('  rel position %s' % ','.join(np.array(rel_position).astype(str)))

        return distance, position, rel_position

    def vraytracing(self, srcs, dests, **kwargs):

        distances = np.zeros(srcs.shape[1])
        positions = np.zeros(srcs.shape)
        rel_positions = np.zeros(srcs.shape)

        for i in range(srcs.shape[1]):
            distances[i], positions[:, i], rel_positions[:, i] = self.raytracing(srcs[:, i], dests[:, i], **kwargs)

        return distances, positions, rel_positions




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
