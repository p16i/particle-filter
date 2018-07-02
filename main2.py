import config
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.image as mpimg

import simple_estimating

import fire
import scenes

import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S',
                    level=config.LOG_LEVEL)

robot_pos = None


def main(scene, no_particles=2000, frame_interval=50, monitor_dpi=218):
    global robot_pos

    scene = getattr(scenes, scene)(no_particles)
    mm = scene.map

    robot_pos = scene.landmarks[0]

    fig = plt.figure()
    ax = plt.axes(xlim=(0, mm.shape[1]), ylim=(0, mm.shape[0]))

    ax.set_yticks([])
    ax.set_xticks([])
    # fig.patch.set_visible(False)
    ax.axis('off')

    background = ax.imshow(mm, origin='lower', cmap='gist_gray', vmax=1, vmin=0)
    overlay = ax.imshow(scene.convolve_mark_overlay, origin='lower', cmap='gist_gray', vmax=1, vmin=0, alpha=0.0)

    no_particles_text = ax.text(0.02, 0.83, '', transform=ax.transAxes)
    time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

    robot, = ax.plot([], [], config.ROBOT_SYMBOL, ms=config.ROBOT_SYMBOL_SIZE)
    particles, = ax.plot([], [], config.PARTICLE_SYMBOL, ms=config.PARTICLE_SYMBOL_SIZE, alpha=config.PARTICLE_OPACITY)

    estimate_model = simple_estimating.position_estimating_model(config.FIELD_DIMS, no_particles)

    sensors = ax.quiver([], [], [], [],
                        scale=1, units='xy', color=config.RADAR_COLOR,
                        headlength=0, headwidth=0,
                        width=config.RADAR_WITDH)

    particle_directions = ax.quiver([], [], [], [],
                                    scale=1, units='xy', color='r', alpha=config.PARTICLE_OPACITY,
                                    width=config.RADAR_WITDH)

    # initialization function: plot the background of each frame
    def init():
        robot.set_data([], [])
        particles.set_data([], [])

        time_text.set_text('')
        no_particles_text.set_text('')

        return background, overlay, robot, particles, time_text, no_particles_text

    # animation function.  This is called sequentially
    def animate(i):
        global robot_pos
        time_text.set_text('Time = %4d' % (i+1))
        no_particles_text.set_text('No. Particles = %d' % no_particles)

        robot_pos, control, v = scene.move(robot_pos)

        radar_src = np.array([[robot_pos[0]]*scene.no_sensors, [robot_pos[1]]*scene.no_sensors])

        radar_theta = scene.radar_thetas + robot_pos[2]
        radar_rel_dest = np.stack(
            (
            np.cos(radar_theta)*config.RADAR_MAX_LENGTH,
            np.sin(radar_theta)*config.RADAR_MAX_LENGTH
            ), axis=0
        )

        radar_dest = np.zeros(radar_rel_dest.shape)
        radar_dest[0, :] = np.clip(radar_rel_dest[0, :] + radar_src[0, :], 0, mm.shape[1])
        radar_dest[1, :] = np.clip(radar_rel_dest[1, :] + radar_src[1, :], 0, mm.shape[0])

        # d = scene.raytracing(robot_pos[:2], radar_dest[:, 3])
        logging.debug('rel dist from analytic')
        logging.debug(radar_rel_dest.T)
        noise_free_measurements, _, radar_rays = scene.vraytracing(radar_src, radar_dest)
        logging.debug('Measurements:')
        logging.debug(noise_free_measurements)

        # todo: sample from motion model

        sensors = ax.quiver(
            radar_src[0, :],
            radar_src[1, :],
            radar_rays[0, :],
            radar_rays[1, :],
            scale=1, units='xy', color=config.RADAR_COLOR, pivot='tail', angles='uv',
            headlength=0, headwidth=0,
            width=config.RADAR_WITDH)

        robot.set_data([robot_pos[0]], [robot_pos[1]])

        # todo: apply controls to particles, with control, and measurement

        # todo: important weighting

        # todo : particle resampling
        particles.set_data(scene.particles[:, 0], scene.particles[:, 1])

        pariticle_directions = ax.quiver(
            scene.particles[:, 0], scene.particles[:, 1],
            np.cos(scene.particles[:, 2])*config.PARTICLE_DIRECTION_DISTANCE,
            np.sin(scene.particles[:, 2])*config.PARTICLE_DIRECTION_DISTANCE,
            scale=1, units='xy', color='r', alpha=config.PARTICLE_OPACITY,
            width=config.RADAR_WITDH)

        return background, overlay, robot, particles, time_text, no_particles_text, sensors, pariticle_directions

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=scene.total_frames, interval=frame_interval, blit=True, repeat=False)

    plt.show()

if __name__ == '__main__':
    fire.Fire(main)



