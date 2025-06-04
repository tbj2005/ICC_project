import numpy as np
import networkx as nx
from math import gcd
from functools import reduce
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


class OpticalNetwork:
    def __init__(self, connectivity_matrix: np.ndarray):
        """Initialize optical network with connectivity matrix
        Args:
            connectivity_matrix: symmetric matrix with zero diagonal,
                               where entry [i,j] is the capacity (Gbps) between node i and j
        """
        if not (connectivity_matrix == connectivity_matrix.T).all():
            raise ValueError("Connectivity matrix must be symmetric")
        if not (connectivity_matrix.diagonal() == 0).all():
            raise ValueError("Diagonal must be zero")

        self.connectivity = connectivity_matrix
        self.num_nodes = connectivity_matrix.shape[0]

    def get_link_capacity(self, node1: int, node2: int) -> float:
        """Get capacity between two nodes (order doesn't matter)"""
        return self.connectivity[node1, node2]

    def get_all_links(self) -> List[Tuple[int, int]]:
        """Get all links as (node1, node2) tuples where node1 < node2"""
        return [(i, j) for i in range(self.num_nodes)
                for j in range(i + 1, self.num_nodes)
                if self.connectivity[i, j] > 0]


class Job:
    def __init__(self,
                 job_id: int,
                 compute_time: float,  # ms
                 comm_time: float,  # ms
                 traffic_matrix: np.ndarray,  # NxN matrix in Gbps
                 placement: List[int] = None):
        """
        Initialize a job with:
        - job_id: unique identifier
        - compute_time: duration of compute phase (ms)
        - comm_time: duration of communication phase (ms)
        - traffic_matrix: NxN matrix showing bandwidth demand between workers
        - placement: list of node indices where workers are placed
        """
        self.job_id = job_id
        self.compute_time = compute_time
        self.comm_time = comm_time
        self.iteration_time = compute_time + comm_time
        self.traffic_matrix = traffic_matrix
        self.placement = placement
        self.num_workers = traffic_matrix.shape[0] if traffic_matrix is not None else 0
        self.time_shift = 0  # Will be set by Cassini scheduler

    def set_placement(self, placement: List[int]):
        """Set the placement of workers for this job"""
        if len(placement) != self.num_workers:
            raise ValueError("Placement size must match number of workers")
        self.placement = placement

    def get_bandwidth_demand(self, link: Tuple[int, int]) -> float:
        """
        Get bandwidth demand on a specific link for this job
        Returns demand in Gbps
        """
        if self.placement is None:
            return 0

        total_demand = 0
        node1, node2 = link

        # Find if this link is used by any worker pair in this job
        for i in range(self.num_workers):
            for j in range(self.num_workers):
                if i == j:
                    continue

                # Check if this worker pair uses the link
                if (self.placement[i] == node1 and self.placement[j] == node2) or \
                        (self.placement[i] == node2 and self.placement[j] == node1):
                    total_demand += self.traffic_matrix[i, j]

        return total_demand


class CassiniScheduler:
    def __init__(self, network: OpticalNetwork):
        self.network = network
        self.jobs = []

    def add_job(self, job: Job):
        """Add a job to be scheduled"""
        self.jobs.append(job)

    def schedule(self):
        """Run the Cassini scheduling algorithm"""
        # Step 1: For each link, find jobs that share it
        link_to_jobs = self._find_competing_jobs()

        # Step 2: For each link with competing jobs, compute compatibility scores
        compatibility_scores = {}
        rotation_angles = {}

        for link, jobs in link_to_jobs.items():
            if len(jobs) > 1:
                score, angles = self._compute_compatibility(link, jobs)
                compatibility_scores[link] = score
                rotation_angles[link] = angles

        # Step 3: Build affinity graph and compute unique time shifts
        affinity_graph = self._build_affinity_graph(link_to_jobs, rotation_angles)
        time_shifts = self._traverse_affinity_graph(affinity_graph)

        # Apply time shifts to jobs
        for job_id, shift in time_shifts.items():
            self.jobs[job_id].time_shift = shift

        return time_shifts, compatibility_scores

    def _find_competing_jobs(self) -> Dict[Tuple[int, int], List[int]]:
        """Find which jobs compete on each link"""
        link_to_jobs = {link: [] for link in self.network.get_all_links()}

        for job_idx, job in enumerate(self.jobs):
            if job.placement is None:
                continue

            # Find all links this job uses
            used_links = set()
            for i in range(job.num_workers):
                for j in range(i + 1, job.num_workers):
                    if job.traffic_matrix[i, j] > 0:
                        link = tuple(sorted((job.placement[i], job.placement[j])))
                        used_links.add(link)

            # Add this job to all links it uses
            for link in used_links:
                link_to_jobs[link].append(job_idx)

        return link_to_jobs

    def _compute_compatibility(self,
                               link: Tuple[int, int],
                               job_indices: List[int]) -> Tuple[float, Dict[int, float]]:
        """
        Compute compatibility score and rotation angles for jobs sharing a link
        Returns (compatibility_score, {job_idx: rotation_angle})
        """
        # Get the jobs
        jobs = [self.jobs[idx] for idx in job_indices]
        link_capacity = self.network.get_link_capacity(*link)

        # Create unified circle with perimeter = LCM of iteration times
        iteration_times = [job.iteration_time for job in jobs]
        unified_perimeter = self._lcm(iteration_times)

        # Create bandwidth demand function for each job on the unified circle
        bw_demands = {}
        for job_idx, job in zip(job_indices, jobs):
            bw_demands[job_idx] = self._create_bw_demand_function(job, unified_perimeter)

        # Find optimal rotation angles to maximize compatibility
        best_score = -np.inf
        best_angles = {}

        # For simplicity, we'll use a grid search over possible angles
        # In practice, you'd use a more sophisticated optimization approach
        angle_steps = np.linspace(0, 2 * np.pi, 36)  # 10 degree steps

        # Try different angle combinations (simplified for this example)
        for base_angle in angle_steps:
            angles = {job_indices[0]: base_angle}  # Fix first job's angle

            # For other jobs, try angles that might interleave
            for i, job_idx in enumerate(job_indices[1:]):
                angles[job_idx] = (base_angle + (i + 1) * np.pi) % (2 * np.pi)

            # Calculate compatibility score
            score = self._calculate_compatibility_score(
                bw_demands, angles, link_capacity, unified_perimeter)

            if score > best_score:
                best_score = score
                best_angles = angles.copy()

        return best_score, best_angles

    def _calculate_compatibility_score(self,
                                       bw_demands: Dict[int, callable],
                                       angles: Dict[int, float],
                                       link_capacity: float,
                                       perimeter: float) -> float:
        """Calculate compatibility score for given rotation angles"""
        num_points = 360  # Number of points to sample around the circle
        excess_sum = 0

        for alpha in np.linspace(0, 2 * np.pi, num_points, endpoint=False):
            total_demand = 0

            for job_idx, angle in angles.items():
                # Get bandwidth demand at this angle considering rotation
                rotated_alpha = (alpha - angle) % (2 * np.pi)
                demand = bw_demands[job_idx](rotated_alpha)
                total_demand += demand

            excess = max(0, total_demand - link_capacity)
            excess_sum += excess

        # Normalize the excess
        avg_excess = excess_sum / num_points
        score = 1 - (avg_excess / link_capacity)

        return score

    def _create_bw_demand_function(self, job: Job, perimeter: float) -> callable:
        """Create bandwidth demand function for a job on the unified circle"""
        num_iterations = int(perimeter / job.iteration_time)

        def bw_demand(alpha: float) -> float:
            """Return bandwidth demand at angle alpha (in radians)"""
            time = (alpha / (2 * np.pi)) * perimeter
            iteration = int(time // job.iteration_time)
            time_in_iteration = time % job.iteration_time

            if time_in_iteration < job.compute_time:
                return 0  # No communication during compute phase
            else:
                # Sum all non-diagonal elements in traffic matrix
                return np.sum(job.traffic_matrix) - np.trace(job.traffic_matrix)

        return bw_demand

    def _build_affinity_graph(self,
                              link_to_jobs: Dict[Tuple[int, int], List[int]],
                              rotation_angles: Dict[Tuple[int, int], Dict[int, float]]) -> nx.Graph:
        """Build the affinity graph connecting jobs to links they use"""
        G = nx.Graph()

        # Add job nodes
        for job in self.jobs:
            G.add_node(f"job_{job.job_id}", type="job")

        # Add link nodes and edges
        for link, job_indices in link_to_jobs.items():
            if len(job_indices) > 1:  # Only consider links with competing jobs
                link_node = f"link_{link[0]}_{link[1]}"
                G.add_node(link_node, type="link")

                # Add edges with rotation angles as weights
                for job_idx in job_indices:
                    angle = rotation_angles[link][job_idx]
                    G.add_edge(f"job_{job_idx}", link_node, weight=angle)

        return G

    def _traverse_affinity_graph(self, G: nx.Graph) -> Dict[int, float]:
        """Traverse affinity graph to compute unique time shifts for all jobs"""
        time_shifts = {}

        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            start_node = next((n for n in subgraph.nodes if subgraph.nodes[n]["type"] == "job"), None)

            if start_node is None:
                continue

            queue = [start_node]
            visited = set([start_node])
            time_shifts[int(start_node.split("_")[1])] = 0  # Set reference job to 0

            while queue:
                current = queue.pop(0)

                if subgraph.nodes[current]["type"] == "job":
                    current_job_id = int(current.split("_")[1])

                    for neighbor in subgraph.neighbors(current):
                        if neighbor not in visited:
                            visited.add(neighbor)

                            for job_neighbor in subgraph.neighbors(neighbor):
                                if job_neighbor != current and job_neighbor not in visited:
                                    visited.add(job_neighbor)
                                    queue.append(job_neighbor)

                                    job_id = int(job_neighbor.split("_")[1])
                                    angle1 = subgraph.edges[current, neighbor]["weight"]
                                    angle2 = subgraph.edges[neighbor, job_neighbor]["weight"]
                                    time_shifts[job_id] = (time_shifts[current_job_id] - angle1 + angle2) % \
                                                          self.jobs[job_id].iteration_time

        return time_shifts

    @staticmethod
    def _lcm(numbers: List[float]) -> float:
        """Calculate LCM of a list of numbers"""

        def _lcm_pair(a, b):
            return a * b // gcd(int(a), int(b))

        return reduce(_lcm_pair, numbers)


def visualize_schedule(jobs: List[Job], duration: float = 500):
    """Visualize the job schedule with time shifts"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, job in enumerate(jobs):
        num_iterations = int(duration / job.iteration_time) + 2
        start_times = [job.time_shift + k * job.iteration_time for k in range(num_iterations)]

        for t in start_times:
            if t < duration:
                # Compute phase
                ax.broken_barh([(t, job.compute_time)],
                               (i - 0.4, 0.8),
                               facecolors='tab:blue',
                               label='Compute' if i == 0 else "")

                # Comm phase
                ax.broken_barh([(t + job.compute_time, job.comm_time)],
                               (i - 0.4, 0.8),
                               facecolors='tab:orange',
                               label='Communication' if i == 0 else "")

    ax.set_yticks(range(len(jobs)))
    ax.set_yticklabels([f"Job {job.job_id}" for job in jobs])
    ax.set_xlabel('Time (ms)')
    ax.set_title('Job Schedule with Cassini Time Shifts')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create a 4-node optical network with 100Gbps links
    conn_matrix = np.array([
        [0, 100, 100, 100],
        [100, 0, 100, 100],
        [100, 100, 0, 100],
        [100, 100, 100, 0]
    ])
    network = OpticalNetwork(conn_matrix)

    # Job 0: All-to-all communication
    traffic0 = np.array([
        [0, 20, 10, 10],
        [20, 0, 10, 10],
        [10, 10, 0, 20],
        [10, 10, 20, 0]
    ])
    job0 = Job(0, compute_time=100, comm_time=50, traffic_matrix=traffic0)
    job0.set_placement([0, 1, 2, 3])  # Place workers on all nodes

    # Job 1: Heavy traffic between specific pairs
    traffic1 = np.array([
        [0, 40, 0, 0],
        [40, 0, 0, 0],
        [0, 0, 0, 40],
        [0, 0, 40, 0]
    ])
    job1 = Job(1, compute_time=150, comm_time=75, traffic_matrix=traffic1)
    job1.set_placement([0, 1, 2, 3])  # Same placement

    # Job 2: Different traffic pattern
    traffic2 = np.array([
        [0, 30, 30, 0],
        [30, 0, 0, 30],
        [30, 0, 0, 30],
        [0, 30, 30, 0]
    ])
    job2 = Job(2, compute_time=120, comm_time=60, traffic_matrix=traffic2)
    job2.set_placement([0, 1, 2, 3])  # Same placement

    # Create scheduler and schedule jobs
    scheduler = CassiniScheduler(network)
    scheduler.add_job(job0)
    scheduler.add_job(job1)
    scheduler.add_job(job2)

    time_shifts, compat_scores = scheduler.schedule()
    print("Time shifts:", time_shifts)
    print("Compatibility scores:", compat_scores)

    visualize_schedule([job0, job1, job2], duration=600)