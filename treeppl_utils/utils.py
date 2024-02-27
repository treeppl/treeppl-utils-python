
import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde
import numpy as np


def find_MAP(values, weights):
    # Normalize the weights to avoid numerical instability
    normalized_weights = np.exp(weights - np.max(weights))

    # Create a KDE using the values and normalized weights
    kde = gaussian_kde(values, weights=normalized_weights)

    # Function to return the negative of KDE (since we want to maximize the KDE)
    def neg_kde(x):
        return -kde(x)[0]
    
    sample_std = np.std(values, ddof=1)  # Sample standard deviation
    bandwidth = kde.factor * sample_std  # Approximate bandwidth used by KDE
    grid_min = min(values) - 3 * bandwidth
    grid_max = max(values) + 3 * bandwidth

    # Find the maximum of the KDE
    result = minimize_scalar(neg_kde, bounds=(grid_min, grid_max), method='bounded')

    if result.success:
        return result.x
    else:
        raise ValueError("Optimization did not converge")
    

def find_mean(values, weights):
    # Normalize the weights to sum up to 1 (to handle any scale of weights)
    normalized_weights = weights / np.sum(weights)

    # Calculate the weighted mean
    weighted_mean = np.sum(values * normalized_weights)

    return weighted_mean


def compute_hdpi(samples, log_weights, hdpi_prob=0.95):
    # Convert log weights to linear scale
    linear_weights = np.exp(log_weights - np.max(log_weights))

    # Perform weighted KDE
    kde = gaussian_kde(samples, weights=linear_weights)

    # Compute the HDPI
    # This step involves finding the densest interval containing hdpi_prob of the probability mass
    # It requires custom implementation, as scipy's gaussian_kde doesn't provide a direct method for this
    
    # Determine the range for the grid points
    sample_std = np.std(samples, ddof=1)  # Sample standard deviation
    bandwidth = kde.factor * sample_std  # Approximate bandwidth used by KDE
    grid_min = min(samples) - 3 * bandwidth
    grid_max = max(samples) + 3 * bandwidth

    # Compute the HDPI
    grid_points = np.linspace(grid_min, grid_max, 1000)

    kde_values = kde(grid_points)
    sorted_indices = np.argsort(kde_values)[::-1]  # Sort by density

    cumulative_prob = 0
    interval_indices = []
    for idx in sorted_indices:
        interval_indices.append(idx)
        cumulative_prob += kde_values[idx] / sum(kde_values)
        if cumulative_prob >= hdpi_prob:
            break

    interval_indices = np.sort(interval_indices)
    hdpi_interval = (grid_points[interval_indices[0]], grid_points[interval_indices[-1]])

    return hdpi_interval


def find_min_hdpi_prob(x, samples, log_weights):
    hdpi_prob = 0.01  # Start with the smallest interval probability
    max_hdpi_prob = 1.00  # Maximum interval probability
    resolution = 0.01  # Increment resolution

    while hdpi_prob <= max_hdpi_prob:
        current_hdpi = compute_hdpi(samples, log_weights, hdpi_prob=hdpi_prob)

        if current_hdpi[0] <= x <= current_hdpi[1]:
            # x is within the interval, return the current interval and its probability
            return hdpi_prob, current_hdpi[0], current_hdpi[1]

        # Increase the interval probability
        hdpi_prob += resolution

    # If the loop completes without returning, x is not in any interval.
    # This is unlikely but could happen if x is an outlier or if there's an issue with the data.
    return None, current_hdpi[0], current_hdpi[1]


def find_min_hdpi_prob_bin(x, samples, log_weights):
    low = 0.01  # Lower bound of the search range
    high = 1.00  # Upper bound of the search range
    resolution = 0.01  # Desired resolution for the search

    while high - low > resolution:
        mid = (high + low) / 2
        current_hdpi = compute_hdpi(samples, log_weights, hdpi_prob=mid)

        if current_hdpi[0] <= x <= current_hdpi[1]:
            # x is within the interval, try a smaller interval
            high = mid
        else:
            # x is not within the interval, try a larger interval
            low = mid

    # Compute the final HDPI at the lower bound of the last search interval
    # This ensures that the HDPI is the tightest one containing x
    final_hdpi = compute_hdpi(samples, log_weights, hdpi_prob=high)
    return high, final_hdpi[0], final_hdpi[1]

def ess(inference_result):
    """
    Calculate the Effective Sample Size (ESS) of the inference result.

    NOTE: This function works only on samples that are either Real 
    (single floating-point numbers) or Real[] (arrays of floating-point numbers)
    due to a limitation in `compress_weights`, see below.
    
    It first compresses the samples by combining those that are identical,
    summing their weights, then calculates the ESS based on these compressed weights.

    Returns:
        The effective sample size as float.
    """
    
    def compress_weights(samples, nweights):
        unique_weights = {}

        # Check if the samples are a list of lists or a simple list
        if samples and isinstance(samples[0], float):
            # Convert each sample to a tuple with a single element
            samples = [(sample,) for sample in samples]

        for sample, weight in zip(samples, nweights):
            sample_tuple = tuple(sample)

            if sample_tuple in unique_weights:
                unique_weights[sample_tuple] += weight
            else:
                unique_weights[sample_tuple] = weight

        # Not needed for now:
        # compressed_samples = [list(sample) for sample in unique_samples.keys()]
        compressed_nweights = np.array(list(unique_weights.values()))

        return compressed_nweights

    def calculate_ess(nweights):
        if nweights is None or len(nweights) == 0:
            return 0

        normalized_weights = nweights / np.sum(nweights)
        sum_of_squares = np.sum(normalized_weights**2)
        return 1 / sum_of_squares

    compressed_nweights = compress_weights(inference_result.samples, inference_result.nweights)
    ess = calculate_ess(compressed_nweights)
    return ess
