import numpy as np
import heapq
from collections import defaultdict


class Business:
    def __init__(self, id, compute_time, data_matrix):
        """
        初始化业务
        :param id: 业务ID
        :param compute_time: 计算时间 (秒)
        :param data_matrix: 数据矩阵 (总pod数×总pod数，对角线为0)
        """
        self.id = id
        self.compute_time = compute_time
        self.data_matrix = data_matrix
        self.iteration_count = 0
        self.total_iteration_time = 0.0
        self.current_state = 'idle'  # 'idle', 'computing', 'communicating'
        self.remaining_time = 0.0
        self.communication_time = self.calculate_communication_time()

    def calculate_communication_time(self):
        """
        根据数据矩阵计算通信时间
        使用矩阵非对角线元素的和作为通信量指标
        """
        # 将对角线置为0（确保不会重复计算）
        np.fill_diagonal(self.data_matrix, 0)
        # 通信时间与非零元素数量和值大小相关
        non_zero_elements = np.count_nonzero(self.data_matrix)
        avg_value = np.sum(self.data_matrix) / non_zero_elements if non_zero_elements > 0 else 0
        return 0.05 * non_zero_elements * avg_value  # 缩放因子

    def start_iteration(self, current_time):
        """开始一次新的迭代"""
        self.iteration_count += 1
        self.current_state = 'computing'
        self.remaining_time = self.compute_time
        return f"Business {self.id} started computing at {current_time:.3f}s"

    def update(self, delta_time):
        """
        更新业务状态
        :return: 如果状态改变，返回状态改变信息，否则返回None
        """
        if self.current_state == 'idle':
            return None

        self.remaining_time -= delta_time
        if self.remaining_time <= 0:
            if self.current_state == 'computing':
                # 计算完成，转为通信状态
                self.current_state = 'communicating'
                self.remaining_time = self.communication_time
                return 'switch_to_communicating'
            elif self.current_state == 'communicating':
                # 通信完成，迭代结束
                self.current_state = 'idle'
                return 'iteration_complete'
        return None


class OpticalNetwork:
    def __init__(self, num_pods, num_ports, recon_time):
        """
        初始化光网络
        :param num_pods: pod总数
        :param num_ports: 端口数
        :param recon_time: 重构时间 (秒)
        """
        self.num_pods = num_pods
        self.num_ports = num_ports
        self.recon_time = recon_time
        self.reconfiguring = False
        self.reconfig_remaining = 0.0
        self.active_business = None  # 当前正在通信的业务

    def can_start_communication(self, business):
        """检查是否可以开始通信（非抢占式，一次只能一个业务通信）"""
        return self.active_business is None

    def start_reconfiguration(self, business):
        """开始网络重构"""
        if not self.reconfiguring and self.active_business is None:
            self.reconfiguring = True
            self.reconfig_remaining = self.recon_time
            self.active_business = business
            return True
        return False

    def start_communication(self, business):
        """开始通信（不经过重构）"""
        if self.active_business is None:
            self.active_business = business
            return True
        return False

    def complete_communication(self):
        """完成通信"""
        self.active_business = None

    def update_reconfiguration(self, delta_time):
        """更新重构状态"""
        if self.reconfiguring:
            self.reconfig_remaining -= delta_time
            if self.reconfig_remaining <= 0:
                self.reconfiguring = False
                return True  # 重构完成
        return False


def simulate(businesses, network, total_simulation_time=100.0):
    """
    运行仿真
    :param businesses: 业务列表
    :param network: 光网络
    :param total_simulation_time: 总仿真时间 (秒)
    :return: 平均迭代时间
    """
    current_time = 0.0
    event_queue = []

    # 初始化所有业务
    for business in businesses:
        heapq.heappush(event_queue, (0.0, 'start_iteration', business.id))

    while current_time < total_simulation_time and event_queue:
        event_time, event_type, business_id = heapq.heappop(event_queue)
        current_time = event_time

        business = next(b for b in businesses if b.id == business_id)

        if event_type == 'start_iteration':
            # 开始计算
            print(business.start_iteration(current_time))
            # 安排计算完成事件
            heapq.heappush(event_queue,
                           (current_time + business.compute_time,
                            'compute_complete',
                            business.id))

        elif event_type == 'compute_complete':
            # 计算完成，尝试开始通信
            if network.can_start_communication(business):
                # 在通信开始前进行网络重构
                if network.start_reconfiguration(business):
                    print(f"Network reconfiguration started at {current_time:.3f}s for business {business.id}")
                    # 安排重构完成事件
                    heapq.heappush(event_queue,
                                   (current_time + network.recon_time,
                                    'recon_complete',
                                    business.id))
                else:
                    # 直接开始通信
                    network.start_communication(business)
                    print(f"Business {business.id} started communication at {current_time:.3f}s")
                    # 安排通信完成事件
                    heapq.heappush(event_queue,
                                   (current_time + business.communication_time,
                                    'communication_complete',
                                    business.id))
            else:
                # 无法立即开始通信，稍后重试
                heapq.heappush(event_queue,
                               (current_time + 0.1,  # 0.1秒后重试
                                'compute_complete',
                                business.id))

        elif event_type == 'recon_complete':
            # 重构完成，开始通信
            print(f"Network reconfiguration completed at {current_time:.3f}s for business {business.id}")
            print(f"Business {business.id} started communication at {current_time:.3f}s")
            heapq.heappush(event_queue,
                           (current_time + business.communication_time,
                            'communication_complete',
                            business.id))

        elif event_type == 'communication_complete':
            # 通信完成，迭代结束
            iteration_time = business.compute_time + business.communication_time
            if network.reconfiguring:
                iteration_time += network.recon_time

            business.total_iteration_time += iteration_time
            network.complete_communication()
            print(f"Business {business.id} completed iteration at {current_time:.3f}s, "
                  f"iteration time: {iteration_time:.3f}s")

            # 安排下一次迭代
            heapq.heappush(event_queue,
                           (current_time + 0.001,  # 微小延迟后开始下一次迭代
                            'start_iteration',
                            business.id))

    # 计算平均迭代时间
    avg_iteration_times = []
    for business in businesses:
        if business.iteration_count > 0:
            avg_time = business.total_iteration_time / business.iteration_count
            avg_iteration_times.append(avg_time)
            print(f"Business {business.id}: {business.iteration_count} iterations, "
                  f"avg iteration time: {avg_time:.3f}s")

    if avg_iteration_times:
        overall_avg = sum(avg_iteration_times) / len(avg_iteration_times)
        print(f"\nOverall average iteration time: {overall_avg:.3f}s")
        return overall_avg
    else:
        print("No iterations completed during simulation")
        return 0.0


def create_data_matrix(num_pods, density=0.3):
    """
    创建数据矩阵（总pod数×总pod数，对角线为0）
    :param num_pods: pod总数
    :param density: 非零元素密度
    :return: 数据矩阵
    """
    matrix = np.random.rand(num_pods, num_pods) * 10
    np.fill_diagonal(matrix, 0)  # 对角线置为0
    # 随机将部分元素置为0
    mask = np.random.rand(num_pods, num_pods) > density
    matrix[mask] = 0
    return matrix


# 示例用法
if __name__ == "__main__":
    # 网络参数
    NUM_PODS = 8
    NUM_PORTS = 16
    RECON_TIME = 0.15  # 150ms重构时间

    # 创建网络
    network = OpticalNetwork(num_pods=NUM_PODS, num_ports=NUM_PORTS, recon_time=RECON_TIME)

    # 创建业务列表
    businesses = []
    np.random.seed(42)

    # 创建5个业务
    for i in range(5):
        compute_time = 0.2 + np.random.rand() * 0.6  # 200-800ms计算时间
        data_matrix = create_data_matrix(NUM_PODS, density=0.4)
        businesses.append(Business(i, compute_time, data_matrix))

    # 运行仿真
    simulate(businesses, network, total_simulation_time=100.0)