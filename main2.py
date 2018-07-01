import config
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.image as mpimg

import simple_estimating

import fire
import scenes

robot_pos = None


def main(scene, no_particles=2000, frame_interval=50, monitor_dpi=218):
    global robot_pos

    scene = getattr(scenes, scene)()
    mm = mpimg.imread(scene.map)
    mm = mm[::-1, :, :]

    robot_pos = scene.landmarks[0]

    fig = plt.figure()
    ax = plt.axes(xlim=(0, mm.shape[1]), ylim=(0, mm.shape[0]))

    ax.set_yticks([])
    ax.set_xticks([])
    # fig.patch.set_visible(False)
    ax.axis('off')

    background = ax.imshow(mm, origin='lower')

    no_particles_text = ax.text(0.02, 0.83, '', transform=ax.transAxes)
    time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

    robot, = ax.plot([], [], config.ROBOT_SYMBOL, ms=config.ROBOT_SYMBOL_SIZE)
    particles, = ax.plot([], [], config.PARTICLE_SYMBOL, ms=config.PARTICLE_SYMBOL_SIZE, alpha=0.5)

    estimate_model = simple_estimating.position_estimating_model(config.FIELD_DIMS, no_particles)

    # initialization function: plot the background of each frame
    def init():
        robot.set_data([], [])
        particles.set_data([], [])

        time_text.set_text('')
        no_particles_text.set_text('')

        return background, robot, particles, time_text, no_particles_text

    # animation function.  This is called sequentially
    def animate(i):
        global robot_pos
        time_text.set_text('Time = %4d' % (i+1))
        no_particles_text.set_text('No. Particles = %d' % no_particles)

        # simulate the robot walk
        # step = [config.ROBOT_STEP_SIZE, np.random.uniform(-config.ROBOT_STEP_THETA_SIZE, config.ROBOT_STEP_THETA_SIZE)]
        # robot_pos = simple_estimating.sample_motion_model(step, robot_pos, config.FIELD_DIMS,
        #                                                   config.ROBOT_MOVEMENT_MODEL_PARAMS)

        robot_pos, control = scene.move(robot_pos)
        # print(robot_pos)


        # create a measure with putting noise on the exact position
        noisey_robo_mes = np.array(robot_pos) + (np.random.normal(0, config.SENSOR_NOISE, len(robot_pos)))
        # do the localization step
        # estimate_model.localization_step(control, noisey_robo_mes)
        #
        # px, py, *_ = estimate_model.get_particle_coordinates()

        robot.set_data([robot_pos[0]], [robot_pos[1]])
        # particles.set_data(px, py)

        return background, robot, particles, time_text, no_particles_text

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=scene.total_frames, interval=frame_interval, blit=True, repeat=False)

    plt.show()

if __name__ == '__main__':
    fire.Fire(main)



