import numpy as np
import config
import matplotlib.image as mpimg
from scipy import signal
from functools import partial
from multiprocessing import Pool



from scipy import stats

import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=config.LOG_LEVEL)

class Environment(object):
    def __init__(self, scene_name, no_particles=20):

        self.scene_name = scene_name
        self.no_particles = no_particles

        map_name = 'scenes/%s.png' % self.scene_name
        self.map = mpimg.imread(map_name)[::-1, :, 0]

        mark = np.ones((config.ROBOT_DIAMETER, config.ROBOT_DIAMETER))

        self.convolve_mark = (signal.convolve2d(1-self.map[:, :], mark, mode='same', boundary='fill', fillvalue=0)) / np.sum(mark)

        self.convolve_mark_overlay = np.copy(self.map)

        threshold = 1/np.sum(mark)
        self.convolve_mark_overlay[self.convolve_mark > threshold] = 0.2

        self.map_with_safe_boundary = np.copy(self.map)
        self.map_with_safe_boundary[self.convolve_mark > threshold] = 0.0 # zero is obstacle.

        self.landmarks = config.SCENCES[scene_name]['landmarks']

        self.controls = Environment._build_control(self.landmarks)

        self.total_frames = len(self.controls)

        self.no_sensors = config.SYSTEM_NO_SENSORS
        self.radar_thetas = (np.arange(0, self.no_sensors) - self.no_sensors // 2)*(np.pi/self.no_sensors)

        logging.info('we have %d controls' % len(self.controls))

        self.traversable_area = np.stack(np.nonzero(1 - (self.map_with_safe_boundary.T < 0.7)), axis=1)

        self.particles = self.uniform_sample_particles()

        self.state_idx = 0

    def uniform_sample_particles(self):
        particles_xy_indices = np.random.choice(self.traversable_area.shape[0], size=self.no_particles, replace=True)
        particles_xy = self.traversable_area[particles_xy_indices]

        particles_theta = np.random.uniform(0.0, 2*np.pi, (self.no_particles, 1))

        return np.hstack([particles_xy, particles_theta])

    def get_control(self):
        control = self.controls[self.state_idx]
        self.state_idx = self.state_idx + 1
        return control

    def perform_control(self, pos, control, noisy_env=True):

        robot_thetha = pos[2] + control[2]
        control = np.array(control)
        theta_control = np.arctan2(control[1], control[0])
        diff_theta = robot_thetha - theta_control

        c, s = np.cos(diff_theta), np.sin(diff_theta)
        rot = np.array(((c, -s), (s, c)))
        vcontrol = rot.dot(control[:2])
        nx = pos[0] + vcontrol[0]
        ny = pos[1] + vcontrol[1]

        if noisy_env:
            nx = nx + np.random.normal(0, config.SYSTEM_MOTION_NOISE[0])
            ny = ny + np.random.normal(0, config.SYSTEM_MOTION_NOISE[1])

        v = (nx - pos[0], ny - pos[1])

        ntheta = np.arctan2(v[1], v[0])

        new_state = (
            nx,
            ny,
            ntheta
        )

        logging.debug('-------')
        logging.debug('control')
        logging.debug(control)
        logging.debug('v')
        logging.debug(v)
        logging.debug('old state')
        logging.debug(pos)
        logging.debug("new state")
        logging.debug(new_state)

        return new_state, v

    def vperform_control(self, vpos, control):
        new_state, new_v = np.zeros(vpos.shape), np.zeros((vpos.shape[0], 2))

        for i in range(vpos.shape[0]):
            new_state[i], new_v[i] = self.perform_control(vpos[i], control)

        return new_state, new_v

    def raytracing(self, src, dest, num_samples=10):
        logging.debug('src %s -> dest %s ' % (','.join(src.astype(str)), ','.join(dest.astype(str))))

        dx = np.where(src[0] < dest[0], 1, -1)
        dy = np.where(src[1] < dest[1], 1, -1)
        x_steps = src[0] + dx*np.linspace(0, np.abs(src[0]-dest[0]), num_samples)
        y_steps = src[1] + dy*np.linspace(0, np.abs(src[1]-dest[1]), num_samples)

        x_steps_int = np.clip(np.round(x_steps).astype(np.int16), 0, self.map.shape[1]-1)
        y_steps_int = np.clip(np.round(y_steps).astype(np.int16), 0, self.map.shape[0]-1)

        mark = np.zeros(self.map.shape)
        mark[y_steps_int, x_steps_int] = 1

        collided_map = self.map[y_steps_int, x_steps_int] < config.SYSTEM_MAP_OCCUPIED_AREA_THRESHOLD

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

    def build_radar_beams(self, pos):
        radar_src = np.array([[pos[0]] * self.no_sensors, [pos[1]] * self.no_sensors])

        radar_theta = self.radar_thetas + pos[2]
        radar_rel_dest = np.stack(
            (
                np.cos(radar_theta)*config.RADAR_MAX_LENGTH,
                np.sin(radar_theta)*config.RADAR_MAX_LENGTH
            ), axis=0
        )

        radar_dest = np.zeros(radar_rel_dest.shape)
        radar_dest[0, :] = np.clip(radar_rel_dest[0, :] + radar_src[0, :], 0, self.map.shape[1])
        radar_dest[1, :] = np.clip(radar_rel_dest[1, :] + radar_src[1, :], 0, self.map.shape[0])

        return radar_src, radar_dest

    def measurement_model(self, pos, observed_measurements):
        if self.map_with_safe_boundary[int(pos[1]), int(pos[0])] < config.SYSTEM_MAP_OCCUPIED_AREA_THRESHOLD:
            return 0.0

        radar_src, radar_dest = self.build_radar_beams(pos)
        noise_free_measurements, _, radar_rays = self.vraytracing(radar_src, radar_dest)

        particle_measurements = noise_free_measurements

        q = 1
        for i in range(particle_measurements.shape[0]):
            q = q * self._measurement_model_p_hit(observed_measurements[i], particle_measurements[i])

        return q

    def vmeasurement_model(self, positions, observed_measurements):
        # return np.ones(positions.shape[0]) / positions.shape[0]

        mm = partial(self.measurement_model, observed_measurements=observed_measurements)

        positions = [positions[i] for i in range(positions.shape[0])]

        with Pool(10) as p:
            weights = p.map(mm, positions)

        weights = np.array(weights)
        total_weights = np.sum(weights)

        if total_weights == 0:
            logging.debug('all weights are zero')
            return False, None
        else:
            return True, weights / total_weights


    @classmethod
    def _measurement_model_p_hit(cls, z, z_star):
        pdf_z = partial(stats.norm.pdf, loc=z_star, scale=config.SYSTEM_MEASURE_MODEL_LOCAL_NOISE_STD)
        prob_z = pdf_z(z)

        mc_grids = np.arange(0, config.RADAR_MAX_LENGTH + config.SYSTEM_MC_INTEGRAL_GRID, config.SYSTEM_MC_INTEGRAL_GRID)

        normalizers = np.sum([pdf_z(x) for x in mc_grids])

        return prob_z / normalizers

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
