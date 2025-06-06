import numpy as np
from math import gcd
from functools import reduce
from collections import defaultdict
import heapq


class CassiniSimulator:
    def __init__(self, num_servers, link_capacity=50):
        """
        Initialize the Cassini simulator for optical interconnected data centers.

        Args:
            num_servers: Number of servers in the data center
            link_capacity: Capacity of each link in Gbps (default 50Gbps)
        """
        self.num_servers = num_servers
        self.link_capacity = link_capacity

        # Create a diagonal connectivity matrix for optical interconnection
        self.connectivity = np.eye(num_servers)
        # Add some cross-connections (optical links between non-adjacent servers)
        for i in range(num_servers):
            for j in range(i + 1, num_servers, 2):  # Connect every other server
                self.connectivity[i, j] = 1
                self.connectivity[j, i] = 1

        # Data structures to track current jobs
        self.active_jobs = []

        # For tracking performance metrics
        self.iteration_times = []

    def add_job(self, job_id, iteration_time, compute_time, comm_bandwidth, traffic_matrix):
        """
        Add a new job to the cluster with its traffic matrix.

        Args:
            job_id: Unique identifier for the job
            iteration_time: Total time for one training iteration (ms)
            compute_time: Duration of compute phase (ms)
            comm_bandwidth: Bandwidth demand during communication phase (Gbps)
            traffic_matrix: NumPy array of shape (num_servers, num_servers)
                            showing communication pattern between servers
        """
        comm_time = iteration_time - compute_time
        job = {
            'id': job_id,
            'iteration_time': iteration_time,
            'compute_time': compute_time,
            'comm_time': comm_time,
            'comm_bandwidth': comm_bandwidth,
            'traffic_matrix': traffic_matrix,
            'time_shift': 0  # Will be set by Cassini scheduler
        }
        self.active_jobs.append(job)

    def find_bottleneck_links(self):
        """
        Find all links (including unidirectional) that have more than one job using them.
        Returns a dict: {(src, dst): [list of jobs sharing this link]}
        """
        # First build link usage: {(src, dst): set(job_ids)}
        link_usage = defaultdict(set)

        for job in self.active_jobs:
            # Get all directed links used by this job (src -> dst)
            srcs, dsts = np.where(job['traffic_matrix'] > 0)
            for src, dst in zip(srcs, dsts):
                link_usage[(src, dst)].add(job['id'])

        # Only keep links shared by multiple jobs
        bottleneck_links = {link: list(jobs) for link, jobs in link_usage.items() if len(jobs) > 1}
        return bottleneck_links

    def lcm(self, numbers):
        """Compute LCM of a list of numbers"""

        def lcm_two(a, b):
            return a * b // gcd(a, b)

        return reduce(lcm_two, numbers, 1)

    def compute_compatibility(self, jobs, link):
        """
        Compute compatibility score and optimal time shifts for jobs sharing a link.

        Args:
            jobs: List of jobs sharing the link
            link: Tuple (server1, server2) identifying the link

        Returns:
            (compatibility_score, time_shifts) where time_shifts is a dict {job_id: shift}
        """
        # Create unified circle with perimeter = LCM of iteration times
        iteration_times = [job['iteration_time'] for job in jobs]
        perimeter = self.lcm(iteration_times)

        # For each job, create its bandwidth demand pattern on the unified circle
        job_patterns = []
        for job in jobs:
            r = job['iteration_time'] / perimeter

            # Create bandwidth demand pattern (0 during compute, bw during comm)
            pattern = np.zeros(360)  # 1-degree precision (360 points)

            # 计算该作业在LCM周期内完整的迭代次数
            num_repeats = int(perimeter / job['iteration_time'])

            # 计算通信阶段在统一圆环上的总角度跨度
            total_comm_angle = 360 * (job['comm_time'] * num_repeats) / perimeter

            # 每次迭代在圆环上的角度跨度
            iter_angle = 360 * job['iteration_time'] / perimeter

            # 每次迭代的通信角度跨度（保持通信/计算比例）
            comm_angle_per_iter = total_comm_angle / num_repeats

            for rep in range(num_repeats):
                iter_start = rep * iter_angle
                comm_start = int(iter_start)
                comm_end = int(iter_start + comm_angle_per_iter)

                # 处理圆环边界
                if comm_end > 360:
                    pattern[comm_start:360] = job['comm_bandwidth']
                    pattern[0:(comm_end % 360)] = job['comm_bandwidth']
                else:
                    pattern[comm_start:comm_end] = job['comm_bandwidth']

            job_patterns.append((job, pattern))

        # Try all possible rotations to find best compatibility
        best_score = -np.inf
        best_shifts = {}

        # We'll use a greedy approach to find good shifts (not exhaustive for performance)
        for base_job, base_pattern in job_patterns:
            # Try aligning other jobs to this base job
            shifts = {base_job['id']: 0}
            total_demand = np.copy(base_pattern)

            for other_job, other_pattern in job_patterns:
                if other_job['id'] == base_job['id']:
                    continue

                # Find shift that minimizes overlap
                min_overlap = np.inf
                best_shift = 0

                # 只需遍历当前作业的迭代角度范围
                for shift in range(int(360 * other_job['iteration_time'] / perimeter)):
                    shifted_pattern = np.roll(other_pattern, shift)
                    overlap = np.sum(np.maximum(total_demand + shifted_pattern - self.link_capacity, 0))

                    if overlap < min_overlap:
                        min_overlap = overlap
                        best_shift = shift

                # 转换最优偏移为时间
                time_shift = (best_shift / 360) * perimeter

                # Convert angle shift to time shift
                # time_shift = (best_shift / 360) * perimeter % other_job['iteration_time']
                shifts[other_job['id']] = time_shift

                # Update total demand
                shifted_pattern = np.roll(other_pattern, best_shift)
                total_demand += shifted_pattern

            # Calculate compatibility score
            excess = np.maximum(total_demand - self.link_capacity, 0)
            score = 1 - np.mean(excess) / self.link_capacity

            if score > best_score:
                best_score = score
                best_shifts = shifts

        return best_score, best_shifts

    def build_affinity_graph(self):
        """Build the affinity graph and compute time shifts for all jobs."""
        # Find all bottleneck links and their shared jobs
        bottleneck_links = self.find_bottleneck_links()

        # Build affinity graph: {job: {link: time_shift}}
        affinity_graph = defaultdict(dict)

        # First compute time shifts for each link independently
        for link, job_ids in bottleneck_links.items():
            job_objects = [j for j in self.active_jobs if j['id'] in job_ids]
            score, shifts = self.compute_compatibility(job_objects, link)

            for job_id, shift in shifts.items():
                affinity_graph[job_id][link] = shift

        # Now traverse the graph to assign unique time shifts
        # We'll use a simple BFS approach as described in the paper
        if not affinity_graph:
            return {}

        # Select a random job as root
        root_job = next(iter(affinity_graph.keys()))
        time_shifts = {root_job: 0}
        queue = [root_job]
        visited = set([root_job])

        while queue:
            current_job = queue.pop(0)

            # Get all links connected to this job
            links = affinity_graph[current_job].keys()

            for link in links:
                # Get all jobs sharing this link
                sharing_jobs = [j['id'] for j in self.active_jobs if
                                j['id'] in affinity_graph and link in affinity_graph[j['id']]]

                for neighbor_job in sharing_jobs:
                    if neighbor_job == current_job:
                        continue

                    if neighbor_job not in visited:
                        # Compute time shift relative to current job
                        t_current = time_shifts[current_job]
                        t_link_current = affinity_graph[current_job][link]
                        t_link_neighbor = affinity_graph[neighbor_job][link]

                        # Find the neighbor job object to get its iteration time
                        neighbor_obj = next(j for j in self.active_jobs if j['id'] == neighbor_job)

                        # The unique time shift formula from the paper
                        t_neighbor = (t_current - t_link_current + t_link_neighbor) % neighbor_obj['iteration_time']

                        time_shifts[neighbor_job] = t_neighbor
                        visited.add(neighbor_job)
                        queue.append(neighbor_job)

        return time_shifts

    def simulate_iteration(self):
        """Simulate one training iteration with refined overlap calculation and non-redundant penalty."""
        # Get time shifts from affinity graph
        time_shifts = self.build_affinity_graph()

        # Apply time shifts to jobs
        for job in self.active_jobs:
            job['time_shift'] = time_shifts.get(job['id'], 0)

        # Precompute LCM period for all jobs' iteration times
        link_penalties = defaultdict(float)  # {link: total_penalty_time}
        lcm_period = self.lcm([j['iteration_time'] for j in self.active_jobs])

        # 第一阶段：计算每条链路的独立惩罚
        bottleneck_links = self.find_bottleneck_links()
        for link, job_ids in bottleneck_links.items():
            jobs = [j for j in self.active_jobs if j['id'] in job_ids]
            if len(jobs) <= 1:
                continue

            # 计算链路总过载数据量
            total_demand = sum(j['traffic_matrix'][link] for j in jobs)
            overload = max(0, total_demand - self.link_capacity)

            # 计算所有作业在该链路上的总重叠通信时间
            total_overlap_time = self.calculate_total_overload(jobs, link, lcm_period)

            # 链路总惩罚 = 过载数据量 / 容量
            link_penalty = (overload / self.link_capacity) * total_overlap_time
            link_penalties[link] = link_penalty

        # 第二阶段：将链路惩罚公平分配到各作业
        job_penalties = defaultdict(float)
        for link, penalty in link_penalties.items():
            jobs = [j for j in self.active_jobs if j['id'] in bottleneck_links[link]]
            total_comm = sum(j['comm_time'] for j in jobs)

            # 按作业通信时间占比分配惩罚
            for job in jobs:
                job_share = job['comm_time'] / total_comm
                job_penalties[job['id']] += penalty * job_share

        # 第三阶段：计算最终迭代时间
        iteration_times = []
        for job in self.active_jobs:
            iteration_time = job['iteration_time'] + job_penalties.get(job['id'], 0)
            iteration_times.append(iteration_time)

        self.iteration_times.append(iteration_times)
        return np.mean(iteration_times)

    def calculate_total_overload(self, jobs, link, lcm_period):
        events = []
        for job in jobs:
            windows = self.get_periodic_windows(
                job['time_shift'],
                job['comm_time'],
                job['iteration_time'],
                lcm_period
            )
            bw = job['traffic_matrix'][link]
            for start, end in windows:
                events.append((start, 'start', bw))
                events.append((end, 'end', bw))

        events.sort()
        active_jobs = set()  # 使用集合防止重复（虽然理论上不应发生）
        current_demand = 0
        total_overload_data = 0
        prev_time = 0

        for time, typ, bw in events:
            # 处理上一时段
            if len(active_jobs) >= 2:
                overload = max(0, current_demand - self.link_capacity)
                total_overload_data += overload * (time - prev_time)

            # 更新当前状态
            if typ == 'start':
                active_jobs.add(bw)
                current_demand += bw
            else:
                active_jobs.discard(bw)
                current_demand -= bw

            prev_time = time

        return total_overload_data / self.link_capacity  # 返回时间惩罚

    def get_periodic_windows(self, start, duration, period, lcm_period):
        """生成周期性通信窗口（支持跨周期边界）"""
        windows = []
        num_repeats = lcm_period // period
        for k in range(num_repeats):
            window_start = (start + k * period) % lcm_period
            window_end = (window_start + duration) % lcm_period
            if window_end < window_start:
                windows.append((window_start, lcm_period))
                windows.append((0, window_end))
            else:
                windows.append((window_start, window_end))
        return windows

    """
    def calculate_windows_overlap(self, windows1, windows2):
        # Calculate total overlap time between two sets of windows.
        overlap = 0

        for (s1, e1) in windows1:
            for (s2, e2) in windows2:
                # Calculate overlap between two individual windows
                overlap += max(0, min(e1, e2) - max(s1, s2))

        return overlap

    def simulate_iteration(self):
        # Simulate one training iteration with current placements and time shifts.
        # Get time shifts from affinity graph
        time_shifts = self.build_affinity_graph()

        # Apply time shifts to jobs
        for job in self.active_jobs:
            job['time_shift'] = time_shifts.get(job['id'], 0)

        # Simulate the iteration for each job
        iteration_times = []

        for job in self.active_jobs:
            # Baseline iteration time (without network congestion)
            base_time = job['iteration_time']

            # Find all bottleneck links this job uses
            bottleneck_links = self.find_bottleneck_links()
            job_links = []

            for link, job_ids in bottleneck_links.items():
                if job['id'] in job_ids:
                    job_links.append(link)

            # Calculate congestion penalty
            congestion_penalty = 0

            for link in job_links:
                # Get all jobs sharing this link
                sharing_jobs = [j for j in self.active_jobs if j['id'] in bottleneck_links[link]]

                # Check if communication phases overlap
                overlaps = 0
                for other_job in sharing_jobs:
                    if other_job['id'] == job['id']:
                        continue

                    # Check if communication phases overlap considering time shifts
                    job1_start = job['time_shift']
                    job1_end = job1_start + job['comm_time']

                    job2_start = other_job['time_shift']
                    job2_end = job2_start + other_job['comm_time']

                    # Check for overlap (simplified)
                    if not (job1_end <= job2_start or job2_end <= job1_start):
                        overlaps += 1

                if overlaps > 0:
                    # Get the actual bandwidth demand for this link
                    i, j = link
                    bw_demand = job['traffic_matrix'][i, j]
                    # Penalty proportional to demand and overlaps
                    congestion_penalty += (bw_demand / self.link_capacity) * job['comm_time'] * overlaps

            # Final iteration time is base time plus congestion penalty
            iteration_time = base_time + congestion_penalty
            iteration_times.append(iteration_time)

        # Record metrics
        self.iteration_times.append(iteration_times)

        return np.mean(iteration_times)
    """
    def run_simulation(self, num_iterations=10):
        """Run the simulation for multiple iterations."""
        avg_times = []
        for _ in range(num_iterations):
            avg_time = self.simulate_iteration()
            avg_times.append(avg_time)
        return avg_times


# Example usage
if __name__ == "__main__":
    # Create a simulator with 24 servers (like in the paper)
    simulator = CassiniSimulator(num_servers=4)

    # Example traffic matrices (in reality these would come from your external source)
    # For VGG16 job: using servers 0-3 with ring allreduce pattern
    vgg16_traffic = np.zeros((4, 4))
    vgg16_traffic[0, 1] = vgg16_traffic[1, 0] = 20  # 20Gbps between server 0-1
    vgg16_traffic[1, 2] = vgg16_traffic[2, 1] = 20  # 20Gbps between server 1-2
    vgg16_traffic[2, 3] = vgg16_traffic[3, 2] = 20  # 20Gbps between server 2-3
    vgg16_traffic[3, 0] = vgg16_traffic[0, 3] = 20  # 20Gbps between server 3-0

    # For ResNet50 job: using servers 4-7 with all-to-all pattern
    resnet_traffic = np.zeros((4, 4))
    for i in range(0, 3):
        for j in range(i + 1, 4):
            resnet_traffic[i, j] = resnet_traffic[j, i] = 45  # 15Gbps between all pairs

    # Add jobs with their traffic matrices
    simulator.add_job("VGG16", 200, 50, 50, vgg16_traffic)
    simulator.add_job("ResNet50", 200, 100, 50, resnet_traffic)

    # Run simulation for 10 iterations
    avg_times = simulator.run_simulation(num_iterations=10)

    print("Average iteration times across jobs for each simulation step:")
    for i, t in enumerate(avg_times):
        print(f"Iteration {i + 1}: {t:.2f} ms")

    print("\nOverall average iteration time:", np.mean(avg_times), "ms")