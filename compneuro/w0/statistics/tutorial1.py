from utils import *


def generate_random_sample(num_points):
    """ Generate a random sample containing a desired number of points (num_points)
  in the range [0, 1] using a random number generator object.

  Args:
    num_points (int): number of points desired in random sample

  Returns:
    dataX, dataY (ndarray, ndarray): arrays of size (num_points,) containing x
    and y coordinates of sampled points

  """
    # Generate desired number of points uniformly between 0 and 1 (using uniform) for
    #     both x and y
    dataX = np.random.uniform(0, 1, num_points)
    dataY = np.random.uniform(0, 1, num_points)

    return dataX, dataY


def generate_random_walk(num_steps, step_size):
    """ Generate the points of a random walk within a 1 X 1 box.

  Args:
    num_steps (int): number of steps in the random walk
    step_size (float): how much each random step size is weighted

  Returns:
    x, y (ndarray, ndarray): the (x, y) locations reached at each time step of the walk

  """
    x = np.zeros(num_steps + 1)
    y = np.zeros(num_steps + 1)

    # Generate the uniformly random x, y steps for the walk
    random_x_steps, random_y_steps = generate_random_sample(num_steps)

    # Take steps according to the randomly sampled steps above
    for step in range(num_steps):
        # take a random step in x and y. We remove 0.5 to make it centered around 0
        x[step + 1] = x[step] + (random_x_steps[step] - 0.5) * step_size
        y[step + 1] = y[step] + (random_y_steps[step] - 0.5) * step_size

        # restrict to be within the 1 x 1 unit box
        x[step + 1] = min(max(x[step + 1], 0), 1)
        y[step + 1] = min(max(y[step + 1], 0), 1)

    return x, y


def show_random_numbers():
    np.random.seed(2)
    num_points = 10
    dataX, dataY = generate_random_sample(num_points)
    plot_random_sample(dataX, dataY)


def show_random_walk():
    np.random.seed(2)
    # Select parameters
    num_steps = 100  # number of steps in random walk
    step_size = 0.5  # size of each step

    # Generate the random walk
    x, y = generate_random_walk(num_steps, step_size)

    # Visualize
    plot_random_walk(x, y, "Rat's location throughout random walk")


def draw_hist():
    # Select parameters for conducting binomial trials
    n = 10
    p = 0.5
    n_samples = 1000

    # Now draw 1000 samples by calling the function again
    left_turn_samples_1000 = np.random.binomial(n, p, size=(n_samples,))

    # Visualize
    count, bins = plot_hist(left_turn_samples_1000, 'Number of left turns in sample')


def draw_poisson():
    # Set random seed
    # np.random.seed(0)

    # Draw 5 samples from a Poisson distribution with lambda = 4
    sampled_spike_counts = np.random.poisson(lam=30, size=1000)
    count, bins = plot_hist(sampled_spike_counts, 'Recorded spikes per second')

    # Print the counts
    print("The samples drawn from the Poisson distribution are " +
          str(sampled_spike_counts))


def my_gaussian(x_points, mu, sigma):
    """ Returns normalized Gaussian estimated at points `x_points`, with
  parameters: mean `mu` and standard deviation `sigma`

  Args:
      x_points (ndarray of floats): points at which the gaussian is evaluated
      mu (scalar): mean of the Gaussian
      sigma (scalar): standard deviation of the gaussian

  Returns:
      (numpy array of floats) : normalized Gaussian evaluated at `x`
  """
    px = 1/(2*np.pi*sigma**2)**1/2 *np.exp(-(x_points-mu)**2/(2*sigma**2))
    # as we are doing numerical integration we have to remember to normalise
    # taking into account the stepsize (0.1)
    px = px / (0.1 * sum(px))

    return px


def draw_gaussian():
    x = np.arange(-8, 9, 0.1)

    # Generate Gaussian
    px = my_gaussian(x, -1, 1)
    my_plot_single(x, px)


if __name__ == '__main__':
    # show_random_numbers()
    # draw_poisson()
    draw_gaussian()

