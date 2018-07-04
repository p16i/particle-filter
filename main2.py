import config
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import fire
import environment

import logging


logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S',
                    level=config.LOG_LEVEL)

robot_pos = None

distance_differences = []


def main(scene, no_particles=2000, total_frames=None, frame_interval=50, show_particles=True, save=False):
    global robot_pos

    scene = environment.Environment(scene, no_particles)
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
    # print(scene.particles.shape)
    particles, = ax.plot(scene.particles[:, 0], scene.particles[:, 1], config.PARTICLE_SYMBOL,
                         ms=config.PARTICLE_SYMBOL_SIZE, alpha=config.PARTICLE_OPACITY)

    sensors = ax.quiver([], [], [], [],
                        scale=1, units='xy', color=config.RADAR_COLOR,
                        headlength=0, headwidth=0,
                        width=config.RADAR_WITDH)

    particle_directions = ax.quiver([], [], [], [], scale=1, units='xy', color='r', width=config.RADAR_WITDH)

    # initialization function: plot the background of each frame
    def init():
        robot.set_data([], [])
        # particles.set_data([], [])

        time_text.set_text('')
        no_particles_text.set_text('')

        return background, overlay, robot, particles, time_text, no_particles_text, particle_directions

    # animation function.  This is called sequentially
    def animate(i):
        # if i < 10:
        #     return background, overlay, robot, particles, time_text, no_particles_text, particle_directions

        # if i % 10 == 0:
        logging.info('>>>>> step %d/%d' % (i+1, scene.total_frames))

        global robot_pos


        time_text.set_text('Time = %4d/%4d' % (i+1, scene.total_frames))
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

        if show_particles:
            particle_positions, particle_velocities = scene.vperform_control(scene.particles, control)

            is_weight_valid, important_weights = scene.vmeasurement_model(particle_positions, noisy_measurements)

            if is_weight_valid:
                # logging.info(','.join(map(lambda x : '%.4f' % x, important_weights)))
                particle_resampling_indicies = np.random.choice(particle_positions.shape[0], particle_positions.shape[0],
                                                                replace=True, p=important_weights)
                particle_resampling = particle_positions[particle_resampling_indicies]
            else:
                particle_resampling = scene.uniform_sample_particles()

            scene.particles = particle_resampling

            particles.set_data(scene.particles[:, 0], scene.particles[:, 1])

            particle_directions.set_UVC(
                np.cos(scene.particles[:, 2])*config.ROBOT_APPROXIMATED_DIRECTION_LENGTH,
                np.sin(scene.particles[:, 2])*config.ROBOT_APPROXIMATED_DIRECTION_LENGTH,
            )

            particle_directions.set_offsets(scene.particles[:, :2])

            approximated_robot_x, approximated_robot_y = np.mean(scene.particles[:, 0]), np.mean(scene.particles[:, 1])
            approximated_robot.set_data([approximated_robot_x], [approximated_robot_y])


            mat = scene.particles - np.array(robot_pos)

            dists = np.sum(np.abs(mat) ** 2, axis=-1) ** (1. / 2)

            mean, std = np.mean(dists), np.std(dists)

            global distance_differences
            distance_differences.append((mean, std))

        return background, overlay, particles, time_text, no_particles_text, robot, sensors, approximated_robot, particle_directions,

    total_frames = scene.total_frames if total_frames is None else total_frames
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=total_frames, interval=frame_interval, blit=True, repeat=False)

    if save:
        name = '%s-%d-particles' % (scene.scene_name, no_particles)
        anim.save('experiments/%s.mp4' % name, fps=10,
                  extra_args=['-vcodec', 'libx264'])

        fig2 = plt.figure()
        ax2 = plt.axes()

        global distance_differences

        distance_differences = np.array(distance_differences)

        x = np.arange(distance_differences.shape[0])
        y = distance_differences[:, 0]
        std = distance_differences[:, 1]
        ax2.plot(x, y)
        ax2.fill_between(x, y - std, y + std, alpha=0.5)

        ax2.set_ylabel("Distance Difference")
        ax2.set_xlabel("Time")
        ax2.set_yticks(np.linspace(0, np.linalg.norm(scene.map.shape[:2])*0.75, 5))
        fig2.savefig('experiments/%s.png' % name)
    else:
        plt.show()

if __name__ == '__main__':
    fire.Fire(main)



