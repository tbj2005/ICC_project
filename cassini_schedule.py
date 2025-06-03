import numpy as np
import matplotlib.pyplot as plt
from math import gcd
from functools import reduce


class Job:
    def __init__(self, name, compute_time, comm_bandwidth, comm_duration):
        self.name = name
        self.compute_time = compute_time  # Duration of compute phase
        self.comm_bandwidth = comm_bandwidth  # Bandwidth demand during communication
        self.comm_duration = comm_duration  # Duration of communication phase
        self.iteration_time = compute_time + comm_duration  # Total iteration time
        self.time_shift = 0  # Initial time shift

    def get_bandwidth_at_time(self, t):
        """Returns bandwidth demand at time t (relative to job's start time)"""
        t_relative = (t - self.time_shift) % self.iteration_time
        if t_relative < self.compute_time:
            return 0  # Compute phase - no bandwidth
        else:
            return self.comm_bandwidth  # Communication phase


def lcm(a, b):
    """Least common multiple of two numbers"""
    return a * b // gcd(a, b)


def lcm_list(numbers):
    """Least common multiple of a list of numbers"""
    return reduce(lcm, numbers)


def calculate_compatibility(jobs, link_capacity, angle_precision=5):
    """
    Calculate compatibility score and optimal time shifts for jobs sharing a link
    using Cassini's geometric abstraction approach.
    """
    # Create unified circle with perimeter = LCM of all iteration times
    unified_perimeter = lcm_list([j.iteration_time for j in jobs])

    # Discretize the circle into angles (0 to 360 degrees)
    angles = np.arange(0, 360, angle_precision)
    angle_rad = np.deg2rad(angles)

    # For each job, map its compute/comm phases to the unified circle
    job_circles = []
    for job in jobs:
        # How many iterations fit in the unified perimeter
        iterations_in_unified = unified_perimeter / job.iteration_time

        # Create bandwidth profile for this job on the unified circle
        bandwidth_profile = np.zeros(len(angles))

        # For each discrete angle, calculate the corresponding time point
        for i, angle in enumerate(angles):
            # Convert angle to time (0 to unified_perimeter)
            t = (angle / 360) * unified_perimeter

            # Find which iteration this time point falls into
            iteration = int(t // job.iteration_time)
            t_in_iteration = t % job.iteration_time

            # Set bandwidth based on phase
            if t_in_iteration < job.compute_time:
                bandwidth_profile[i] = 0  # Compute phase
            else:
                bandwidth_profile[i] = job.comm_bandwidth  # Comm phase

        job_circles.append({
            'job': job,
            'bandwidth_profile': bandwidth_profile,
            'iterations_in_unified': iterations_in_unified
        })

    # Optimization to find best rotation angles (time shifts)
    best_score = -np.inf
    best_shifts = [0] * len(jobs)

    # Try different rotation combinations (simplified search)
    # In practice, this would use a more sophisticated optimization approach
    for shift_attempt in range(0, 360, angle_precision):
        test_shifts = [shift_attempt * i for i in range(len(jobs))]

        # Calculate total bandwidth at each angle
        total_bandwidth = np.zeros(len(angles))
        for i, jc in enumerate(job_circles):
            shifted_profile = np.roll(jc['bandwidth_profile'],
                                      int(test_shifts[i] / angle_precision))
            total_bandwidth += shifted_profile

        # Calculate compatibility score
        excess_bandwidth = np.maximum(total_bandwidth - link_capacity, 0)
        avg_excess = np.mean(excess_bandwidth)
        score = 1 - (avg_excess / link_capacity)

        if score > best_score:
            best_score = score
            best_shifts = test_shifts

    # Convert rotation angles to time shifts
    for i, jc in enumerate(job_circles):
        rotation_angle = best_shifts[i]
        time_shift = (rotation_angle / 360) * unified_perimeter
        # Mod by iteration time to keep shift within one iteration
        time_shift = time_shift % jc['job'].iteration_time
        jc['job'].time_shift = time_shift

    return best_score


def simulate_jobs(jobs, link_capacity, duration):
    """
    Simulate job execution over time with their time shifts
    and plot bandwidth utilization.
    """
    time_points = np.arange(0, duration, 0.1)
    total_bandwidth = np.zeros(len(time_points))
    individual_bandwidths = {}

    for job in jobs:
        bw = np.array([job.get_bandwidth_at_time(t) for t in time_points])
        individual_bandwidths[job.name] = bw
        total_bandwidth += bw

    # Plot results
    plt.figure(figsize=(12, 6))

    # Plot individual job bandwidths
    for name, bw in individual_bandwidths.items():
        plt.plot(time_points, bw, label=name, alpha=0.7)

    # Plot total bandwidth
    plt.plot(time_points, total_bandwidth, 'k-', label='Total', linewidth=2)

    # Plot link capacity
    plt.axhline(y=link_capacity, color='r', linestyle='--', label='Link Capacity')

    plt.xlabel('Time (ms)')
    plt.ylabel('Bandwidth (Gbps)')
    plt.title('Network Bandwidth Utilization with Cassini Scheduling')
    plt.legend()
    plt.grid(True)
    plt.show()

    return total_bandwidth


# Example usage
if __name__ == "__main__":
    # Create some jobs with different compute/comm patterns
    job1 = Job("VGG16", compute_time=150, comm_bandwidth=40, comm_duration=100)
    job2 = Job("ResNet50", compute_time=200, comm_bandwidth=30, comm_duration=50)
    job3 = Job("BERT", compute_time=180, comm_bandwidth=25, comm_duration=70)

    jobs = [job1, job2, job3]
    link_capacity = 50  # Gbps

    # Calculate compatibility and optimal time shifts
    score = calculate_compatibility(jobs, link_capacity)
    print(f"Compatibility score: {score:.2f}")
    for job in jobs:
        print(f"{job.name} time shift: {job.time_shift:.1f} ms")

    # Simulate and plot the bandwidth usage
    total_bw = simulate_jobs(jobs, link_capacity, duration=500)

    # Calculate congestion metrics
    congestion = np.sum(total_bw > link_capacity) / len(total_bw)
    print(f"Percentage of time with congestion: {congestion * 100:.1f}%")
