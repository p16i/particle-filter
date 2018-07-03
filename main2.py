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


def main(scene, no_particles=2000, frame_interval=50, monitor_dpi=218, save=False):
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
    overlay = ax.imshow(scene.convolve_mark_overlay, origin='lower', cmap='gist_gray', vmax=1, vmin=0, alpha=0.1)

    no_particles_text = ax.text(0.02, 0.83, '', transform=ax.transAxes)
    time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

    robot, = ax.plot([], [], config.ROBOT_SYMBOL, ms=config.ROBOT_SYMBOL_SIZE)
    approximated_robot, = ax.plot([], [], config.ROBOT_APPROXIMATED_SYMBOL, ms=config.ROBOT_SYMBOL_SIZE,
                                  alpha=config.PARTICLE_OPACITY)
    particles, = ax.plot([], [], config.PARTICLE_SYMBOL, ms=config.PARTICLE_SYMBOL_SIZE, alpha=config.PARTICLE_OPACITY)

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
        if i % 10 == 0:
            logging.info('step %d' % i)

        global robot_pos

        time_text.set_text('Time = %4d' % (i+1))
        no_particles_text.set_text('No. Particles = %d' % no_particles)

        control = scene.get_control()
        robot_pos, _ = scene.perform_control(robot_pos, control)

        radar_src, radar_dest = scene.build_radar_beams(robot_pos)

        noise_free_measurements, _, radar_rays = scene.vraytracing(radar_src, radar_dest)
        logging.debug('Measurements:')
        logging.debug(noise_free_measurements)

        noisy_measurements = noise_free_measurements + np.random.normal(0, config.RADAR_NOISE_STD, noise_free_measurements.shape[0])

        sensors.set_UVC(radar_rays[0, :], radar_rays[1, :])
        sensors.set_offsets(radar_src.T)


        robot.set_data([robot_pos[0]], [robot_pos[1]])

        particle_positions, particle_velocities = scene.vperform_control(scene.particles, control)

        is_weight_valid, important_weights = scene.vmeasurement_model(particle_positions, noisy_measurements)

        if is_weight_valid:
            particle_resampling_indicies = np.random.choice(particle_positions.shape[0], particle_positions.shape[0],
                                                            replace=True, p=important_weights)
            particle_resampling = particle_positions[particle_resampling_indicies]
        else:
            particle_resampling = scene.uniform_sample_particles()

        scene.particles = particle_resampling

        particles.set_data(scene.particles[:, 0], scene.particles[:, 1])

        particle_directions.set_UVC(
            np.cos(scene.particles[:, 2])*config.PARTICLE_DIRECTION_DISTANCE,
            np.sin(scene.particles[:, 2])*config.PARTICLE_DIRECTION_DISTANCE,
        )

        particle_directions.set_offsets(scene.particles[:, :2])

        approximated_robot.set_data([np.mean(scene.particles[:, 0])], [np.mean(scene.particles[:, 1])])

        return background, overlay, particles, time_text, no_particles_text, robot, sensors, particle_directions, approximated_robot

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=scene.total_frames-1, interval=frame_interval, blit=True, repeat=False)

    if save:
        anim.save('experiments/%s-%d-particles.mp4' % (scene.scene_name, no_particles), fps=15,
                  extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()

if __name__ == '__main__':
    fire.Fire(main)



