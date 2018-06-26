import config
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import simple_estimating

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-config.MAP_WIDTH, config.MAP_WIDTH), ylim=(-config.MAP_HEIGHT, config.MAP_HEIGHT))

ax.grid()

no_particles_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

robot, = ax.plot([], [], config.ROBOT_SYMBOL, ms=config.ROBOT_SYMBOL_SIZE)
particles, = ax.plot([], [], config.PARTICLE_SYMBOL, ms=config.PARTICLE_SYMBOL_SIZE, alpha=0.5)
sensors, = ax.plot([], [], config.SENSOR_SYMBOL, ms=config.SENSOR_SYMBOL_SIZE)

robot_pos = [config.ROBOT_DEFAULT_X, config.ROBOT_DEFAULT_Y, config.ROBOT_DEFAULT_THETA]

m = simple_estimating.position_estimating_model(config.FIELD_DIMS, config.NO_PARTICLES)

# initialization function: plot the background of each frame
def init():
    robot.set_data([], [])
    particles.set_data([], [])

    sensors.set_data([], [])

    time_text.set_text('')
    no_particles_text.set_text('')

    return robot, particles, time_text, sensors, no_particles_text

# animation function.  This is called sequentially
def animate(i):
    time_text.set_text('Time = %4d' % (i+1))
    no_particles_text.set_text('No. Particles = %d' % config.NO_PARTICLES)

    sensors.set_data(config.SENSOR_POSITIONS_X, config.SENSOR_POSITIONS_Y)


    # simulate the robot walk
    global robot_pos
    step = [config.ROBOT_STEP_SIZE, np.random.uniform(-config.ROBOT_STEP_THETA_SIZE, config.ROBOT_STEP_THETA_SIZE)]
    robot_pos = simple_estimating.sample_motion_model(step, robot_pos, config.FIELD_DIMS,
                                                      config.ROBOT_MOVEMENT_MODEL_PARAMS)


    # create a measure with putting noise on the exact position
    noisey_robo_mes = np.array(robot_pos) + (np.random.normal(0, config.SENSOR_NOISE, len(robot_pos)))
    # do the localization step
    m.localization_step(step, noisey_robo_mes)

    px, py, *_ = m.get_particle_coordinates()

    robot.set_data([robot_pos[0]], [robot_pos[1]])
    particles.set_data(px, py)

    return particles, robot, time_text, sensors, no_particles_text


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=config.TOTAL_FRAMES, interval=config.INTERVAL_PER_FRAME, blit=True, repeat=False)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
