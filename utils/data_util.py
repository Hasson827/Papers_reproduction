import numpy as np

def checkerboard(N: int = 10000, x_min: float = -4, x_max: float = 4, 
                y_min: float = -4, y_max: float = 4, resolution: int = 100) -> np.ndarray:
    """
    Generate the checkerboard pattern samples.
    Args:
        N (int): Number of samples to generate.
        x_min (float): Minimum x-coordinate.
        x_max (float): Maximum x-coordinate.
        y_min (float): Minimum y-coordinate.
        y_max (float): Maximum y-coordinate.
        resolution (int): Resolution of the grid.
    Returns:
        data: (N, 2) array of sampled points.
    """
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    length = 4
    checkerboard = np.indices((length, length)).sum(axis=0) % 2

    sampled_points = []
    while len(sampled_points) < N:
        x_sample = np.random.uniform(x_min, x_max)
        y_sample = np.random.uniform(y_min, y_max)
        
        i = int((x_sample - x_min) / (x_max - x_min) * length)
        j = int((y_sample - y_min) / (y_max - y_min) * length)
        
        if checkerboard[j, i] == 1:
            sampled_points.append((x_sample, y_sample))

    sampled_points = np.array(sampled_points)
    return sampled_points

def blobs(N: int = 10000, centers: list = None, std: float = 0.5) -> np.ndarray:
    """
    Generate Gaussian blob-like samples.
    Args:
        N (int): Number of samples to generate.
        centers (list): List of center points for the blobs.
        std (float): Standard deviation for the Gaussian noise.
    Returns:
        data: (N, 2) array of sampled points.
    """
    if centers is None:
        centers = [(-2, -2), (2, 2), (-2, 2), (2, -2)]

    k = len(centers)
    points = []
    for i in range(N):
        c = centers[np.random.randint(0, k)]
        x, y = np.random.normal(c, std, size=(2,))
        points.append((x, y))

    return np.array(points)

def moons(N: int = 10000, r: float = 1.0, noise: float = 0.1) -> np.ndarray:
    """
    Generate two interleaving half circles (moons).
    Args:
        N (int): Number of samples to generate.
        r (float): Radius of the moons.
        noise (float): Standard deviation of the Gaussian noise.
    Returns:
        data: (N, 2) array of sampled points.
    """
    N2 = N // 2
    theta1 = np.random.rand(N2) * np.pi
    theta2 = np.random.rand(N2) * np.pi + np.pi
    x1 = r * np.cos(theta1) + noise * np.random.randn(N2)
    y1 = r * np.sin(theta1) + noise * np.random.randn(N2)
    x2 = r * np.cos(theta2) + noise * np.random.randn(N2) + r
    y2 = r * np.sin(theta2) + noise * np.random.randn(N2) - r
    return np.vstack([np.stack([x1, y1], axis=1), 
                      np.stack([x2, y2], axis=1)])

def circles(N: int = 10000, radius: tuple = None, noise: float = 0.05):
    """
    Generate Concentric Circles with different radii.
    Args:
        N (int): Number of samples to generate.
        radius (tuple): Radii of the circles.
        noise (float): Standard deviation of the Gaussian noise.
    Returns:
        data: (N, 2) array of sampled points.
    """
    points = []
    k = len(radius)
    for i in range(N):
        ri = radius[i % k]
        theta = np.random.rand() * 2 * np.pi
        x = ri * np.cos(theta) + noise * np.random.randn()
        y = ri * np.sin(theta) + noise * np.random.randn()
        points.append((x, y))
    return np.array(points)

def spiral(N: int = 10000, a: float = 0.5, b: float = 0.2, noise: float = 0.05) -> np.ndarray:
    """
    Generate a spiral pattern.
    Args:
        N (int): Number of samples to generate.
        a (float): Spiral parameter.
        b (float): Spiral parameter.
        noise (float): Standard deviation of the Gaussian noise.
    Returns:
        data: (N, 2) array of sampled points.
    """
    theta = np.linspace(0, 4 * np.pi, N)
    r = a + b * theta
    x = r * np.cos(theta) + noise * np.random.randn(N)
    y = r * np.sin(theta) + noise * np.random.randn(N)
    return np.stack([x, y], axis=1)

def pinwheel(N: int = 10000, k: int = 5, r: float = 1.0, noise: float = 0.1, rate: float = 0.25):
    """
    Generate a pinwheel pattern.
    Args:
        N (int): Number of samples to generate.
        k (int): Number of "spokes" in the pinwheel.
        r (float): Radius of the pinwheel.
        noise (float): Standard deviation of the Gaussian noise.
        rate (float): Rate of rotation for the pinwheel.
    Returns:
        data: (N, 2) array of sampled points.
    """
    points = []
    per = N // k
    for i in range(k):
        theta = 2 * np.pi * i / k
        for _ in range(per):
            radial = np.random.randn() * noise + r
            angle = theta + rate * radial
            x = radial * np.cos(angle)
            y = radial * np.sin(angle)
            points.append((x, y))
    return np.array(points)

def generate_original_distribution(N: int = 10000, distribution_type: str = "blobs", **kwargs) -> np.ndarray:
    """
    Generate the original distribution based on the specified type.
    Args:
        N (int): Number of samples to generate.
        distribution_type (str): Type of distribution to generate.
        **kwargs: Additional arguments for the specific distribution function.
    Returns:
        data: (N, 2) array of sampled points.
    """
    if distribution_type == "blobs":
        return blobs(N, **kwargs)
    elif distribution_type == "moons":
        return moons(N, **kwargs)
    elif distribution_type == "circles":
        return circles(N, **kwargs)
    elif distribution_type == "spiral":
        return spiral(N, **kwargs)
    elif distribution_type == "pinwheel":
        return pinwheel(N, **kwargs)
    elif distribution_type == "checkerboard":
        return checkerboard(N, **kwargs)
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
