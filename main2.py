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
angle_differences = []


def main(scene, no_particles=10, total_frames=None, frame_interval=50, show_particles=True, save=False,
         no_random_particles=0):
    global robot_pos

    scene = environment.Environment(scene, no_particles)
    mm = scene.map

    robot_pos = scene.paths[0][0]

    fig = plt.figure()
    ax = plt.axes(xlim=(0, mm.shape[1]), ylim=(0, mm.shape[0]))

    ax.set_yticks([0, mm.shape[0]])
    ax.set_xticks([0, mm.shape[1]])
    # fig.patch.set_visible(False)
    # ax.axis('off')

    background = ax.imshow(mm, origin='lower', cmap='gist_gray', vmax=1, vmin=0)
    overlay = ax.imshow(scene.convolve_mark_overlay, origin='lower', cmap='gist_gray', vmax=1, vmin=0, alpha=0.1)

    no_particles_text = ax.text(0.02, 0.83, '', transform=ax.transAxes)
    time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

    robot, = ax.plot([], [], config.ROBOT_SYMBOL, ms=config.ROBOT_SYMBOL_SIZE)
    approximated_robot, = ax.plot([], [], config.ROBOT_APPROXIMATED_SYMBOL, ms=config.ROBOT_SYMBOL_SIZE,
                                  alpha=config.PARTICLE_OPACITY)

    particles, = ax.plot(scene.particles[:, 0], scene.particles[:, 1], config.PARTICLE_SYMBOL,
                         ms=config.PARTICLE_SYMBOL_SIZE, alpha=config.PARTICLE_OPACITY)
    particle_directions = ax.quiver([], [], [], [], scale=1, units='xy', color='r', width=config.RADAR_WITDH)

    sensors = ax.quiver([], [], [], [],
                        scale=1, units='xy', color=config.RADAR_COLOR,
                        headlength=0, headwidth=0,
                        width=config.RADAR_WITDH)

    view_random_particles, = ax.plot([], [], 'ko', ms=config.PARTICLE_SYMBOL_SIZE,
                                alpha=config.PARTICLE_OPACITY)
    view_random_particle_directions = ax.quiver([], [], [], [], scale=1, units='xy', color='k', width=config.RADAR_WITDH)


    def init():
        robot.set_data([], [])

        time_text.set_text('')
        no_particles_text.set_text('')

        return background, overlay, robot, particles, time_text, no_particles_text, particle_directions

    def animate(i):
        global robot_pos, distance_differences, angle_differences

        logging.info('>>>>> step %d/%d' % (scene.total_move+1, scene.total_frames))

        time_text.set_text('Time = %4d/%4d' % (scene.total_move+1, scene.total_frames))

        particle_label = 'No. Particles = %d' % no_particles
        if no_random_particles > 0:
            particle_label = '%s + %d' % (particle_label, no_random_particles)
        no_particles_text.set_text(particle_label)

        teleport_pos, control = scene.get_control()

        if teleport_pos:
            robot_pos = teleport_pos
        else:
            robot_pos, _ = scene.perform_control(robot_pos, control)

            radar_src, radar_dest = scene.build_radar_beams(robot_pos)

            noise_free_measurements, _, radar_rays = scene.vraytracing(radar_src, radar_dest)
            logging.debug('Measurements:')
            logging.debug(noise_free_measurements)

            noisy_measurements = noise_free_measurements + np.random.normal(0, config.RADAR_NOISE_STD, noise_free_measurements.shape[0])

            sensors.set_UVC(radar_rays[0, :], radar_rays[1, :])
            sensors.set_offsets(radar_src.T)

            if show_particles:
                particle_positions, particle_velocities = scene.vperform_control(scene.particles, control)

                is_weight_valid, important_weights = scene.vmeasurement_model(particle_positions, noisy_measurements)

                if is_weight_valid:
                    # logging.info(','.join(map(lambda x : '%.4f' % x, important_weights)))
                    particle_resampling_indicies = np.random.choice(particle_positions.shape[0], particle_positions.shape[0],
                                                                    replace=True, p=important_weights)
                    particle_resampling = particle_positions[particle_resampling_indicies]
                else:
                    particle_resampling = scene.uniform_sample_particles(no_particles)

                scene.particles = particle_resampling

                particles.set_data(scene.particles[:, 0], scene.particles[:, 1])

                position_differences = scene.particles[:, :2] - np.array(robot_pos)[:2]

                dists = np.sum(np.abs(position_differences) ** 2, axis=-1) ** (1. / 2)

                distance_differences.append((np.mean(dists), np.std(dists)))

                angle_diffs = np.abs(scene.particles[:, 2] - robot_pos[2]).reshape(-1, 1)
                angle_diffs = np.hstack((angle_diffs, 2*np.pi - angle_diffs))
                angle_diffs = np.min(angle_diffs, axis=1)
                angle_differences.append((np.mean(angle_diffs), np.std(angle_diffs)))

                approximated_robot_x, approximated_robot_y = np.mean(scene.particles[:, 0]), np.mean(scene.particles[:, 1])
                approximated_robot.set_data([approximated_robot_x], [approximated_robot_y])

                particle_directions.set_UVC(
                    np.cos(scene.particles[:, 2])*config.ROBOT_APPROXIMATED_DIRECTION_LENGTH,
                    np.sin(scene.particles[:, 2])*config.ROBOT_APPROXIMATED_DIRECTION_LENGTH,
                    )
                particle_directions.set_offsets(scene.particles[:, :2])

                if no_random_particles > 0 and (i + 1) % 10 == 0:
                    random_particles = scene.uniform_sample_particles(no_random_particles)

                    view_random_particles.set_data(random_particles[:, 0], random_particles[:, 1])
                    view_random_particle_directions.set_UVC(
                        np.cos(random_particles[:, 2])*config.ROBOT_APPROXIMATED_DIRECTION_LENGTH,
                        np.sin(random_particles[:, 2])*config.ROBOT_APPROXIMATED_DIRECTION_LENGTH,
                        )
                    view_random_particle_directions.set_offsets(random_particles[:, :2])

                    scene.particles = np.vstack((scene.particles, random_particles))
                else:
                    view_random_particles.set_data([], [])
                    view_random_particle_directions.set_UVC([], [])
                    view_random_particle_directions.set_offsets([])

        robot.set_data([robot_pos[0]], [robot_pos[1]])

        return background, overlay, particles, time_text, no_particles_text, robot, sensors, approximated_robot, \
               particle_directions, view_random_particles, view_random_particle_directions

    total_frames = scene.total_frames if total_frames is None else total_frames
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=total_frames, interval=frame_interval, blit=True, repeat=False)

    if save:
        name = '%s-%d-particles' % (scene.scene_name, no_particles)

        if no_random_particles > 0:
            name = '%s-with-rp-%d' % (name, no_random_particles)

        anim.save('experiments/%s.mp4' % name, fps=10,
                  extra_args=['-vcodec', 'libx264'])

        fig2, res_ax = plt.subplots(nrows=2, ncols=1, sharex=True)

        global distance_differences, angle_differences

        distance_differences = np.array(distance_differences)
        angle_differences = np.array(angle_differences)

        x = np.arange(distance_differences.shape[0])
        y = distance_differences[:, 0]
        std = distance_differences[:, 1]
        res_ax[0].plot(x, y)
        res_ax[0].fill_between(x, y - std, y + std, alpha=0.5)

        y_tick_positions = np.linspace(0, np.linalg.norm(scene.map.shape[:2])*0.75, 5)
        res_ax[0].set_ylabel("Position Difference")
        res_ax[0].set_yticks(y_tick_positions)

        x = np.arange(angle_differences.shape[0])
        y = angle_differences[:, 0]
        std = angle_differences[:, 1]
        res_ax[1].plot(x, y)
        res_ax[1].fill_between(x, y - std, y + std, alpha=0.5)

        y_tick_thetas = [0, np.pi]
        res_ax[1].set_ylabel("Angle Difference")
        res_ax[1].set_xlabel("Time")
        res_ax[1].set_yticks(y_tick_thetas)
        res_ax[1].set_yticklabels(['0', '$\pi$'])

        if scene.kidnapping_occur_at:
            res_ax[0].plot([scene.kidnapping_occur_at]*2, [y_tick_positions[0], y_tick_positions[-1]], 'r--')
            res_ax[1].plot([scene.kidnapping_occur_at]*2, [y_tick_thetas[0], y_tick_thetas[-1]], 'r--')


        title = '%d particles' % no_particles

        if no_random_particles > 0:
            title = '%s\n (with %d random particles every 10 steps)' % (title, no_random_particles)

        fig2.suptitle(title)

        fig2.savefig('experiments/%s.png' % name)
    else:
        plt.show()

if __name__ == '__main__':
    fire.Fire(main)



