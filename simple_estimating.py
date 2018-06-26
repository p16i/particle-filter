import numpy as np
from scipy.stats import norm as normdb
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

class position_estimating_model(object):
    """Position estimating model class"""
    def __init__(self, field_dim, particle_nr):
        # dimension of the field
        self.field_dim = field_dim
        # number of particles
        self.particle_nr = particle_nr

        # model parameters used for the motion model
        self.theta_var = 0.03
        self.position_std = 0.7

        # create initial particles uniformly distributed on the field
        # particles of the form [x,y,theta]
        # (x,y) are the coordinates, theta is the angel of the robot looking direction
        self.X = list(zip(np.random.uniform(0,field_dim[0],particle_nr),
                          np.random.uniform(0,field_dim[1],particle_nr),
                          np.random.uniform(0,2*np.pi,particle_nr)))

    def localization_step(self, control_data, environment_measurement):
        """ Makes a MCM-localization step with all particles. """
        # in X_new_prior there are the particles after the step
        # in X_new there are the chosen particles
        # todo : reimplement with numpy vector
        X_new = []
        X_new_prior = [[],[]]
        # do the motion with everey particle (x) with likelihood (w)
        for particle in self.X:
            x = sample_motion_model(control_data, particle, self.field_dim, [self.theta_var, self.position_std])
            w = measurement_model(environment_measurement, x, self.field_dim)
            X_new_prior[0].append(x)
            X_new_prior[1].append(w)

        # normalize the w's to get a discrete distribution of the index set of the particles
        w_sum = sum(X_new_prior[1])
        prop = [i / w_sum for i in X_new_prior[1]]

        # choose particle_nr new particles
        for m in range(self.particle_nr):
            # choose random index from the particle index set with the discrete distribution from the w's
            i = np.random.choice(list(range(self.particle_nr)), p=prop)
            X_new.append(X_new_prior[0][i])
        #save the new particles
        self.X = X_new

    def get_particle_coordinates(self):

        x = np.array(self.X)

        return x[:, 0], x[:, 1], np.cos(x[:, 2]), np.sin(x[:, 2])

    def plot(self, exact_position):
        """ Create a plot of the particles, the true position and the mean of the particles """

        particle_x, particle_y, particle_dir_x, particle_dir_y = self.get_particle_coordinates()

        # limit the plot to the field
        plt.axis([0, self.field_dim[0], 0, self.field_dim[1]])
        # plot particles position and direction
        plt.scatter(particle_x, particle_y, c=[1]*self.particle_nr, s= [1]*self.particle_nr, alpha=0.5)
        plt.quiver(particle_x, particle_y, particle_dir_x, particle_dir_y, angles="xy", pivot="mid", color="violet")

        # Get mean position, direction and the exact position, direction
        mean_x, mean_y = np.mean(particle_x), np.mean(particle_y)
        mean_theta = np.mean([p[2] for p in self.X])
        mean_dir_x = np.cos(mean_theta)
        mean_dir_y = np.sin(mean_theta)
        exact_dir_x = np.cos(exact_position[2])
        exact_dir_y = np.sin(exact_position[2])

        # plot mean value in red
        plt.scatter(mean_x, mean_y, s=40, c="red")
        plt.quiver(mean_x, mean_y, mean_dir_x, mean_dir_y, angles="xy", pivot="mid", color="red")
        # plot exact value in green
        plt.scatter(exact_position[0], exact_position[1] , s=40, c="green")
        plt.quiver(exact_position[0], exact_position[1], exact_dir_x, exact_dir_y, angles="xy", pivot="mid", color="green")
        # show plot
        plt.show()

def sample_motion_model(control_data, particle, field_dim, normal_param):
    """ Very simple sample motion model. It chooses sample from a truncated normal around the predicted new location.

    control_data: pair of the form (v,w) with stepsize v and turning angle w
    particle: position and direction of the particle
    field_dim: dimension of the field where the particles should lie
    normal_param: parameters for the movement (how good can the robot translate the command into reality?)
    """

    # Control data and particle state
    [v, w] = control_data
    [x,y,theta] = particle
    [var, std] = normal_param

    # New angle
    theta = theta + w + np.random.normal(scale = var)
    # New expected position
    x_exp = x + v*np.cos(theta)
    y_exp = y + v*np.sin(theta)

    # The new sample should lie in the field and be sample from a normal with mean (x_exp, y_exp)
    # for the chose of a and b : https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    [a, b] = [- x_exp / std, (field_dim[0] - x_exp) / std ]
    x = truncnorm.rvs(a,b, loc=x_exp, scale=std)
    [a, b] = [- y_exp / std, (field_dim[1] - y_exp) / std ]
    y = truncnorm.rvs(a,b, loc=y_exp, scale=std)
    # return new position and direction

    return [x,y,theta]


def measurement_model(environment_measurement, particle, field_dim):
    """ Gives the the likelihood p(z|x) with measurement z and particle state (position, direction) x

    environment_measurement: noisey estimations of the x and y position
    particle:  position of the particle"""

    # expected variance of the noise from the measurement data
    var = 1;
    var_ang = 0.52

    # measurements of the position and the angle
    [x_m, y_m, theta_m] = environment_measurement
    # state of the particle
    [x, y, theta] = particle

    # likelihood p(z|x)
    prop = normdb.pdf(x_m, loc=x, scale = var) * normdb.pdf(y_m, loc=y, scale = var) * normdb.pdf(theta_m, loc=theta, scale = var_ang)
    return prop
