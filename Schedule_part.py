import copy
import numpy as np

GPU_usage = 0.8
GPU_flops = 9.7


def mountain_climb(job_set):
    """
    爬山法迭代最优解
    :param job_set: 业务六元组集合（业务索引，业务参数量，batch size，通信模式， 并行度， 单条数据 token 量）, 通信模式分 ring, ps,
    all_to_all 三类，分别使用0， 1， 2 表示
    :return:
    """
    a = [job_set[i][0] for i in range(0, len(job_set))]
    b = []


def topology_deploy(group_a, group_b, job_set, cluster_pod, unit_gpu):
    """
    计算每组的 OXC 拓扑，并求解对应的单迭代时长
    :param group_a: a 组业务索引
    :param group_b: b 组业务索引
    :param job_set: 业务六元组集合
    :param cluster_pod: 集群中的 pod 数目
    :param unit_gpu: 单 pod gpu 数目
    :return:
    """
    t_a = [train_single_server(job_set[i][2], job_set[i][1], GPU_flops, GPU_usage, job_set[i][5]) for i in group_a]
    t_b = [train_single_server(job_set[i][2], job_set[i][1], GPU_flops, GPU_usage, job_set[i][5]) for i in group_b]
    ps_a = [i for i in group_a if job_set[i][3] == 1]
    ps_b = [i for i in group_b if job_set[i][3] == 1]
    train_a_min, gpu_num_a = count_train_time(t_a, cluster_pod * unit_gpu - ps_b)
    train_b_min, gpu_num_b = count_train_time(t_b, cluster_pod * unit_gpu - ps_a)
    group_traffic, sum_traffic = traffic_count(job_set)


def deploy_server(group_index, unit_gpu_num, job_set, group_traffic_single_link, group_traffic_size, pod, pod_gpu):
    """
    各组并行单元的部署方案生成，产生拓扑
    :return:
    """
    reverse_gpu = np.array([pod_gpu for i in range(0, pod)])
    output_traffic = np.zeros(pod)
    group_traffic_size_copy = copy.deepcopy(group_traffic_size)
    link_bool_matrix = np.zeros([pod, pod], dtype=bool)
    traffic_single_job = np.empty(len(group_index), dtype=None)
    while np.max(group_traffic_size_copy) > 0:
        index = np.argmax(group_traffic_size_copy)
        node = []
        traffic_matrix = np.zeros([pod, pod])
        if job_set[group_index[index][3]] == 0:
            for i in range(0, job_set[group_index[index]][4]):
                count_gpu = unit_gpu_num + 0
                k = 0
                rank_node = -1
                while count_gpu > 0:
                    pod_index = np.argmin(output_traffic)
                    reverse_gpu[pod_index] -= min(count_gpu, reverse_gpu[pod_index])
                    if k == 0:
                        output_traffic[pod_index] += group_traffic_single_link
                        node.append(pod_index)
                        rank_node = pod_index
                    else:
                        output_traffic[pod_index] += count_gpu * group_traffic_single_link / unit_gpu_num
                    count_gpu -= min(count_gpu, reverse_gpu[pod_index])
                    k += 1




def count_train_time(t_list, gpu_available):
    """
    计算用满所有 GPU 的平均时间和分配方式
    :param t_list: 单 GPU 训练时间队列
    :param gpu_available: 可用 GPU 数目
    :return:
    """
    t_train = sum(t_list) / gpu_available
    gpu_assign = [t_list[i] / t_train for i in range(0, len(t_list))]
    return t_train, gpu_assign


def train_single_server(batch_size, model_parameter, flops, usage, data_token):
    """
    业务单 GPU 训练用时
    :param batch_size: 每次迭代的数据量
    :param model_parameter: 业务参数量
    :param flops: GPU 峰值 flops
    :param usage: GPU 利用率
    :param data_token: 单条数据平均 token 量
    :return:
    """
    return 8 * data_token * batch_size * model_parameter / (flops * usage)


def traffic_count(job_set):
    """
    根据业务种类，参数量和 batch size 计算通信流量
    :param job_set:
    :return:
    """
    traffic = []
    sum_output_traffic = []
    for i in range(0, len(job_set)):
        if job_set[i][3] == 0:
            dp = job_set[i][4]
            traffic.append(2 * (dp - 1) * job_set[i][1] / dp)
            sum_output_traffic.append(2 * (dp - 1) * job_set[i][1])
        if job_set[i][3] == 1:
            dp = job_set[i][4]
            traffic.append(job_set[i][1])
            sum_output_traffic.append(2 * dp * job_set[i][1])
        if job_set[i][3] == 2:
            ep = job_set[i][4]
            traffic.append((job_set[i][2] * job_set[i][5]) / ep)
            sum_output_traffic.append((ep - 1) * job_set[i][2] * job_set[i][5])
    return traffic, sum_output_traffic
