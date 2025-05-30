import numpy as np
import networkx as nx
from math import gcd
from functools import reduce
from itertools import combinations
import matplotlib.pyplot as plt


def lcm(a, b):
    """计算两个数的最小公倍数"""
    return a * b // gcd(a, b)


def lcm_list(numbers):
    """计算数字列表的最小公倍数"""
    return reduce(lcm, numbers, 1)


class Job:
    def __init__(self, name, iteration_time, communication_pattern):
        """
        初始化一个具有通信模式的作业

        参数:
            name: 作业标识符
            iteration_time: 一个训练迭代的时间(毫秒)
            communication_pattern: 元组列表(start_time, duration, bandwidth)
                                  表示一个迭代中的通信阶段
        """
        self.name = name
        self.iteration_time = iteration_time
        self.communication_pattern = communication_pattern
        self.time_shift = 0  # 将由调度器设置

    def get_bandwidth_at_time(self, t):
        """
        获取时间t的带宽需求(相对于迭代开始)
        """
        t = (t - self.time_shift) % self.iteration_time
        for start, duration, bw in self.communication_pattern:
            if start <= t < start + duration:
                return bw
        return 0  # 计算阶段

    def __repr__(self):
        return f"Job({self.name}, iter={self.iteration_time}ms)"


class Link:
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity
        self.jobs = []

    def add_job(self, job):
        self.jobs.append(job)

    def compute_compatibility(self, angle_precision=5):
        """
        使用Cassini的几何抽象计算共享此链接的作业的兼容性分数

        返回:
            compatibility_score: 介于0(不兼容)和1(完全兼容)之间
            time_shifts: 字典{job: optimal_time_shift}
        """
        if len(self.jobs) < 2:
            return 1.0, {job: 0 for job in self.jobs}

        # 创建统一圆，周长为迭代时间的最小公倍数
        iteration_times = [job.iteration_time for job in self.jobs]
        unified_perimeter = lcm_list(iteration_times)

        # 离散化角度(简化优化)
        angles = np.linspace(0, 2 * np.pi, num=360 // angle_precision)

        best_score = -np.inf
        best_time_shifts = {}

        # 简化优化:尝试不同的旋转组合
        # (实际实现会使用论文表1中的优化公式)
        for rotations in self._generate_rotation_combinations():
            time_shifts = {}
            excess_sum = 0

            # 应用旋转并计算超额带宽
            for i, job in enumerate(self.jobs):
                # 将旋转角度转换为时间偏移
                rotation_angle = rotations[i]
                time_shift = (rotation_angle / (2 * np.pi)) * unified_perimeter
                time_shift = time_shift % job.iteration_time
                time_shifts[job] = time_shift

            # 检查每个角度的带宽
            for alpha in angles:
                total_bw = 0
                for job in self.jobs:
                    # 将角度转换为统一圆中的时间
                    t = (alpha / (2 * np.pi)) * unified_perimeter
                    # 获取作业在此时间的带宽(考虑时间偏移)
                    job_t = (t - time_shifts[job]) % job.iteration_time
                    total_bw += job.get_bandwidth_at_time(job_t)

                excess = max(0, total_bw - self.capacity)
                excess_sum += excess / self.capacity  # 归一化

            # 计算兼容性分数
            avg_excess = excess_sum / len(angles)
            score = 1 - avg_excess

            if score > best_score:
                best_score = score
                best_time_shifts = time_shifts

        return best_score, best_time_shifts

    def _generate_rotation_combinations(self, steps=8):
        """生成可能的旋转组合(为演示简化)"""
        # 实际实现会替换为适当的优化
        angles = np.linspace(0, 2 * np.pi, steps, endpoint=False)
        for combo in combinations(angles, len(self.jobs)):
            yield combo


class CassiniScheduler:
    def __init__(self):
        self.jobs = []
        self.links = []
        self.affinity_graph = nx.Graph()

    def add_job(self, job):
        self.jobs.append(job)
        self.affinity_graph.add_node(job.name, type='job', obj=job)

    def add_link(self, link):
        self.links.append(link)
        self.affinity_graph.add_node(link.name, type='link', obj=link)

    def connect_job_to_link(self, job_name, link_name):
        """在亲和图中连接作业和链接"""
        self.affinity_graph.add_edge(job_name, link_name)

    def schedule(self):
        """
        执行Cassini调度:
        1. 计算链路级兼容性和时间偏移
        2. 构建亲和图
        3. 遍历图以分配唯一时间偏移
        """
        # 步骤1: 计算链路级时间偏移
        link_time_shifts = {}  # {link: {job: time_shift}}

        for link in self.links:
            score, shifts = link.compute_compatibility()
            print(f"链路 {link.name} 兼容性分数: {score:.2f}")
            link_time_shifts[link] = shifts

            # 更新亲和图边权重
            for job, t_shift in shifts.items():
                if self.affinity_graph.has_edge(job.name, link.name):
                    self.affinity_graph.edges[job.name, link.name]['weight'] = t_shift

        # 步骤2: 使用亲和图遍历分配唯一时间偏移
        # (论文中的算法1)
        job_time_shifts = self._traverse_affinity_graph()

        # 将时间偏移应用到作业
        for job in self.jobs:
            if job.name in job_time_shifts:
                job.time_shift = job_time_shifts[job.name]
                print(f"为 {job.name} 分配时间偏移: {job.time_shift:.1f}ms")
            else:
                job.time_shift = 0

        return job_time_shifts

    def _traverse_affinity_graph(self):
        """亲和图的BFS遍历以分配唯一时间偏移"""
        time_shifts = {}
        visited = set()

        # 分别处理每个连通组件
        for component in nx.connected_components(self.affinity_graph):
            # 找到一个作业节点开始BFS
            start_node = next(n for n in component
                              if self.affinity_graph.nodes[n]['type'] == 'job')

            # 用起始作业初始化队列(时间偏移=0)
            queue = [start_node]
            time_shifts[start_node] = 0
            visited.add(start_node)

            while queue:
                current = queue.pop(0)

                # 访问所有邻居
                for neighbor in self.affinity_graph.neighbors(current):
                    if neighbor in visited:
                        continue

                    node_type = self.affinity_graph.nodes[neighbor]['type']
                    edge_data = self.affinity_graph.edges[current, neighbor]

                    if node_type == 'link':
                        # 遍历 job -> link: 减去边权重
                        link = self.affinity_graph.nodes[neighbor]['obj']
                        for job in link.jobs:
                            if job.name != current and job.name not in visited:
                                # 计算此作业的时间偏移
                                t_shift = (time_shifts[current] - edge_data['weight']) % job.iteration_time
                                time_shifts[job.name] = t_shift
                                visited.add(job.name)
                                queue.append(job.name)

        return time_shifts

    def visualize_communication(self, duration=500):
        """可视化带有时间偏移的作业通信模式"""
        plt.figure(figsize=(12, 6))

        for i, job in enumerate(self.jobs):
            times = np.arange(0, duration, 1)
            bandwidths = [job.get_bandwidth_at_time(t) for t in times]

            plt.plot(times, bandwidths, label=f"{job.name} (偏移={job.time_shift:.1f}ms)")

        plt.xlabel('时间 (毫秒)')
        plt.ylabel('带宽需求')
        plt.title('带有时间偏移的作业通信模式')
        plt.legend()
        plt.grid(True)
        plt.show()


# 示例用法
if __name__ == "__main__":
    # 创建一些具有不同通信模式的作业
    # 格式: (开始时间, 持续时间, 带宽)
    job1 = Job("VGG16", 255, [(0, 141, 45)])  # 下行阶段: 0-141ms, 45Gbps
    job2 = Job("VGG19", 255, [(141, 114, 45)])  # 上行阶段: 141-255ms, 45Gbps
    job3 = Job("GPT-2", 400, [(0, 100, 30), (200, 100, 30)])  # 两个通信阶段

    # 创建具有容量(论文中为50Gbps)的链路
    link1 = Link("链路1", 50)
    link1.add_job(job1)
    link1.add_job(job2)  # 这两个应该是兼容的

    link2 = Link("链路2", 50)
    link2.add_job(job1)
    link2.add_job(job3)  # 兼容性较差

    # 创建调度器并设置
    scheduler = CassiniScheduler()
    scheduler.add_job(job1)
    scheduler.add_job(job2)
    scheduler.add_job(job3)
    scheduler.add_link(link1)
    scheduler.add_link(link2)

    # 在亲和图中连接作业和链路
    scheduler.connect_job_to_link("VGG16", "链路1")
    scheduler.connect_job_to_link("VGG19", "链路1")
    scheduler.connect_job_to_link("VGG16", "链路2")
    scheduler.connect_job_to_link("GPT-2", "链路2")

    # 执行调度
    scheduler.schedule()

    # 可视化通信模式
    scheduler.visualize_communication()