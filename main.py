import numpy as np
from simple_estimating import *

# field dimension and parameters the real robot should have
field_dim = [20,10];
normal_param = [0.01, 0.5]

# model
m = position_estimating_model([20,10], 2000)

# the way of the robot, first value is the stepsize, second value is the turning angle (-pi/2 means 90° clockwise)
robo_walk = [ [5,0] , [5,-1/2*np.pi], [5,-1/2*np.pi] , [5,-1/2*np.pi] , [5,-1/2*np.pi] , [5,-1/2*np.pi],
			[5,-1/2*np.pi], [5,-1/2*np.pi] , [5,-1/2*np.pi] , [5,-1/2*np.pi] , [5,-1/2*np.pi] ]
# robo starting position
robo_pos = [10, 8, 0]

# let the robo walk and estimate the position
for robo_step in robo_walk:
    # simulate the robo walk with sample_motion_model
    robo_pos = sample_motion_model(robo_step, robo_pos, field_dim, normal_param)
    # create a measure with putting noise on the exact position
    noisey_robo_mes = np.array(robo_pos) + (np.random.normal(0,0.3,3))
    # do the localization step
    m.localization_step(robo_step, noisey_robo_mes)
    # plot the result
    m.plot(robo_pos)
    print('xxx')
