import numpy as np
import networkx as nx
from math import gcd
from functools import reduce
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt


def lcm(a, b):
    """Compute least common multiple of two numbers"""
    return a * b // gcd(a, b)


def lcm_list(numbers):
    """Compute LCM of a list of numbers"""
    return reduce(lcm, numbers, 1)


class Job:
    def __init__(self, job_id: int, iteration_time: float, phases: List[Tuple[float, float, float]]):
        """
        重构后的Job类，使用流量大小(GB)替代带宽需求

        Args:
            job_id: 作业唯一标识
            iteration_time: 迭代总时间(ms)
            phases: 通信阶段列表，每个阶段为(start_time, duration, data_volume)
                   start_time: 阶段开始时间(ms)
                   duration: 阶段持续时间(ms)
                   data_volume: 传输数据量(GB)
        """
        self.job_id = job_id
        self.iteration_time = iteration_time
        self.phases = phases
        self.time_shift = 0  # 时间偏移量

    def get_bandwidth_at_time(self, t: float) -> float:
        """
        根据流量大小和持续时间计算瞬时带宽需求(Gbps)
        公式: 带宽(Gbps) = 数据量(GB) * 8 / 持续时间(ms)
        """
        t_shifted = (t - self.time_shift) % self.iteration_time
        for start, duration, data_vol in self.phases:
            if start <= t_shifted < start + duration:
                return (data_vol * 8) / (duration / 1000)  # GB->Gb, ms->s
        return 0

    def get_phase_info_at_time(self, t: float) -> Tuple[float, float]:
        """获取当前时间的阶段信息和原始流量大小"""
        t_shifted = (t - self.time_shift) % self.iteration_time
        for start, duration, data_vol in self.phases:
            if start <= t_shifted < start + duration:
                return (data_vol, duration)
        return (0, 0)


class OpticalTopology:
    """拓扑类保持不变"""

    def __init__(self, connectivity_matrix: np.ndarray):
        self.connectivity = connectivity_matrix
        self.num_nodes = connectivity_matrix.shape[0]
        self.links = self._create_links()

    def _create_links(self) -> Dict[Tuple[int, int], Dict]:
        links = {}
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.connectivity[i, j] > 0:
                    links[(i, j)] = {
                        'capacity': self.connectivity[i, j] * 10,  # 每条光连接10Gbps
                        'jobs': set()
                    }
        return links

    def add_job_path(self, job_id: int, path: List[int]):
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            if (src, dst) in self.links:
                self.links[(src, dst)]['jobs'].add(job_id)

    def remove_job_path(self, job_id: int, path: List[int]):
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            if (src, dst) in self.links and job_id in self.links[(src, dst)]['jobs']:
                self.links[(src, dst)]['jobs'].remove(job_id)


class CassiniScheduler:
    def __init__(self, topology: OpticalTopology):
        self.topology = topology
        self.jobs = {}
        self.placement_candidates = []

    def add_job(self, job: Job, path: List[int]):
        self.jobs[job.job_id] = job
        self.topology.add_job_path(job.job_id, path)

    def remove_job(self, job_id: int, path: List[int]):
        if job_id in self.jobs:
            del self.jobs[job_id]
        self.topology.remove_job_path(job_id, path)

    def generate_placement_candidates(self, num_candidates: int = 5):
        """生成候选布局(与之前相同)"""
        self.placement_candidates = []
        for _ in range(num_candidates):
            candidate = {}
            for job_id, job in self.jobs.items():
                if np.random.random() < 0.3:
                    path_length = np.random.randint(2, 4)
                    path = [np.random.randint(0, self.topology.num_nodes) for _ in range(path_length)]
                else:
                    path = self._get_current_path(job_id)
                candidate[job_id] = path
            self.placement_candidates.append(candidate)

    def _get_current_path(self, job_id: int) -> List[int]:
        return [0, 1, 2]  # 简化的默认路径

    def evaluate_compatibility(self, candidate: Dict[int, List[int]]) -> Tuple[float, Dict[int, float]]:
        """评估候选布局的兼容性"""
        temp_topology = OpticalTopology(self.topology.connectivity.copy())
        for job_id, path in candidate.items():
            temp_topology.add_job_path(job_id, path)

        affinity_graph = self._build_affinity_graph(temp_topology)
        time_shifts = self._compute_time_shifts(affinity_graph)

        # 应用时间偏移
        original_shifts = {job_id: job.time_shift for job_id, job in self.jobs.items()}
        for job_id, shift in time_shifts.items():
            self.jobs[job_id].time_shift = shift

        # 计算链路兼容性
        link_scores = []
        for link, link_data in temp_topology.links.items():
            if len(link_data['jobs']) >= 2:
                score = self._compute_link_compatibility(link, link_data['jobs'])
                link_scores.append(score)

        # 恢复原始偏移
        for job_id, shift in original_shifts.items():
            self.jobs[job_id].time_shift = shift

        avg_score = np.mean(link_scores) if link_scores else 0
        return avg_score, time_shifts

    def _build_affinity_graph(self, topology: OpticalTopology) -> nx.Graph:
        """构建Affinity图(与之前相同)"""
        G = nx.Graph()
        for job_id in self.jobs:
            G.add_node(f"j{job_id}", bipartite=0, type="job")
        for link, link_data in topology.links.items():
            if len(link_data['jobs']) >= 2:
                link_name = f"l{link[0]}-{link[1]}"
                G.add_node(link_name, bipartite=1, type="link", capacity=link_data['capacity'])
                for job_id in link_data['jobs']:
                    G.add_edge(f"j{job_id}", link_name, weight=0)
        return G

    def _compute_link_compatibility(self, link: Tuple[int, int], job_ids: set) -> float:
        """基于流量大小的兼容性计算"""
        jobs = [self.jobs[jid] for jid in job_ids]
        link_capacity = self.topology.links[link]['capacity']
        iteration_times = [job.iteration_time for job in jobs]
        unified_perimeter = lcm_list(iteration_times)
        num_angles = 360  # 1度步长

        best_score = -np.inf
        best_rotations = {}
        angle_steps = np.linspace(0, 2 * np.pi, 36, endpoint=False)  # 10度间隔

        for rot_angles in np.array(np.meshgrid(*[angle_steps] * len(jobs))).T.reshape(-1, len(jobs)):
            max_demand = 0
            total_excess = 0

            for alpha in np.linspace(0, 2 * np.pi, num_angles, endpoint=False):
                total_bw = 0
                for job, rot in zip(jobs, rot_angles):
                    t = (alpha / (2 * np.pi)) * unified_perimeter
                    t_job = (t - (rot / (2 * np.pi)) * job.iteration_time) % job.iteration_time

                    # 动态计算带宽需求
                    data_vol, duration = job.get_phase_info_at_time(t_job)
                    if duration > 0:
                        bw = (data_vol * 8) / (duration / 1000)  # GB->Gb, ms->s
                        total_bw += bw

                excess = max(0, total_bw - link_capacity)
                total_excess += excess

            avg_excess = total_excess / num_angles
            score = 1 - (avg_excess / link_capacity)

            if score > best_score:
                best_score = score
                best_rotations = {job.job_id: rot for job, rot in zip(jobs, rot_angles)}

        # 转换旋转角度为时间偏移
        for job_id, rot in best_rotations.items():
            time_shift = (rot / (2 * np.pi)) * unified_perimeter % self.jobs[job_id].iteration_time
            self.jobs[job_id].time_shift = time_shift

        return best_score

    def _compute_time_shifts(self, affinity_graph: nx.Graph) -> Dict[int, float]:
        """计算时间偏移(与之前相同)"""
        time_shifts = {}
        for component in nx.connected_components(affinity_graph):
            subgraph = affinity_graph.subgraph(component)
            job_nodes = [n for n in subgraph.nodes if subgraph.nodes[n]['type'] == 'job']
            if not job_nodes: continue

            start_node = job_nodes[0]
            queue = [start_node]
            visited = set([start_node])
            time_shifts[int(start_node[1:])] = 0

            while queue:
                current = queue.pop(0)
                for neighbor in subgraph.neighbors(current):
                    if neighbor in visited: continue

                    if subgraph.nodes[neighbor]['type'] == 'link':
                        link_capacity = subgraph.nodes[neighbor]['capacity']
                        job_ids = [int(n[1:]) for n in subgraph.neighbors(neighbor)
                                   if subgraph.nodes[n]['type'] == 'job']
                        iteration_times = [self.jobs[jid].iteration_time for jid in job_ids]
                        unified_perimeter = lcm_list(iteration_times)

                        for jid in job_ids:
                            if jid not in time_shifts:
                                time_shifts[jid] = jid * unified_perimeter / (
                                            len(job_ids) * self.jobs[jid].iteration_time)
                                queue.append(f"j{jid}")
                                visited.add(f"j{jid}")
        return time_shifts

    def schedule(self):
        """调度主流程(与之前相同)"""
        if not self.jobs: return

        self.generate_placement_candidates(num_candidates=5)
        best_score, best_candidate, best_shifts = -np.inf, None, {}

        for candidate in self.placement_candidates:
            score, time_shifts = self.evaluate_compatibility(candidate)
            if score > best_score:
                best_score, best_candidate, best_shifts = score, candidate, time_shifts

        for job_id, shift in best_shifts.items():
            self.jobs[job_id].time_shift = shift

        return best_candidate, best_shifts, best_score

    def plot_utilization(self, link: Tuple[int, int]):
        """增强的可视化：显示带宽和流量大小"""
        job_ids = self.topology.links[link]['jobs']
        if not job_ids:
            print(f"No jobs share link {link}")
            return

        unified_perimeter = lcm_list([self.jobs[jid].iteration_time for jid in job_ids])
        time_points = np.linspace(0, unified_perimeter, 500)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # 绘制带宽需求
        for jid in job_ids:
            job = self.jobs[jid]
            bandwidth = [job.get_bandwidth_at_time(t) for t in time_points]
            ax1.plot(time_points, bandwidth, label=f"Job {jid}")
        ax1.axhline(y=self.topology.links[link]['capacity'], color='r', linestyle='--', label='Link Capacity')
        ax1.set_ylabel("Bandwidth (Gbps)")
        ax1.set_title(f"Bandwidth Utilization on Link {link}")
        ax1.legend()
        ax1.grid()

        # 绘制流量大小
        for jid in job_ids:
            job = self.jobs[jid]
            data_volumes = [job.get_phase_info_at_time(t)[0] for t in time_points]
            ax2.plot(time_points, data_volumes, label=f"Job {jid}")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Data Volume (GB)")
        ax2.set_title("Data Volume Transmission")
        ax2.legend()
        ax2.grid()

        plt.tight_layout()
        plt.show()


# 示例使用
if __name__ == "__main__":
    # 4节点光互连拓扑
    topology_matrix = np.array([
        [0, 2, 1, 0],  # 节点0到节点1有2条连接，到节点2有1条
        [1, 0, 0, 1],  # 节点1到节点0和3各有1条连接
        [0, 0, 0, 2],  # 节点2到节点3有2条连接
        [0, 0, 0, 0]  # 节点3无出向连接
    ])

    topology = OpticalTopology(topology_matrix)
    scheduler = CassiniScheduler(topology)

    # 定义作业(迭代时间ms, [(开始时间ms, 持续时间ms, 数据量GB)])
    jobs = [
        Job(1, 200, [(0, 50, 10), (100, 50, 5)]),  # 作业1: 两个阶段(10GB和5GB)
        Job(2, 300, [(0, 100, 20), (150, 50, 15)]),  # 作业2: 两个阶段(20GB和15GB)
        Job(3, 250, [(50, 75, 12), (150, 50, 8)]),  # 作业3: 两个阶段(12GB和8GB)
    ]

    paths = [
        [0, 1, 3],  # 作业1路径
        [0, 2, 3],  # 作业2路径
        [1, 3]  # 作业3路径
    ]

    for job, path in zip(jobs, paths):
        scheduler.add_job(job, path)

    # 运行调度器
    best_candidate, best_shifts, best_score = scheduler.schedule()

    print("最优候选布局:")
    for job_id, path in best_candidate.items():
        print(f"作业{job_id}: 路径{path}")

    print("\n最优时间偏移:")
    for job_id, shift in best_shifts.items():
        print(f"作业{job_id}: {shift:.2f} ms 偏移")

    print(f"\n最优兼容性得分: {best_score:.2f}")

    # 可视化链路利用率
    print("\n链路(0,1)利用率分析:")
    scheduler.plot_utilization((0, 1))