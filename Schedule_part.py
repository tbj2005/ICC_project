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
    unit_gpu_a = [int(gpu_num_a[i] / job_set[group_a[i]][4]) for i in range(0, len(group_a))]
    deploy_server(group_a, unit_gpu_a, job_set, group_traffic, sum_traffic, )


def deploy_server(group, unit_gpu_num, job_set, group_traffic_single_link, group_traffic_size, pod, pod_gpu
                  , oxc_num, oxc_tor_bandwidth_rate):
    """
    各组并行单元的部署方案生成，产生拓扑，注意连接选择全双工，默认使用 INC
    :return:
    """
    reverse_gpu = np.array([pod_gpu for i in range(0, pod)])
    # output_traffic = np.zeros(pod)
    group_traffic = [group_traffic_size[i] for i in group]
    group_traffic_copy = copy.deepcopy(group_traffic)
    undeploy_job = []
    link_bool_matrix = np.zeros([pod, pod])
    # link_each_job = np.empty(len(job_set), dtype=None)
    output_oxc = np.zeros(pod)
    output_tor = np.zeros(pod)
    oxc_matrix = np.zeros([pod, pod])
    while np.max(group_traffic_copy) < 0:
        job_index = group[np.argmax(group_traffic_copy)]
        group_traffic_copy[np.argmax(group_traffic_copy)] = -1
        single_output_oxc = np.zeros(pod)
        single_output_tor = np.zeros(pod)
        single_oxc_matrix = np.zeros([pod, pod])
        if job_set[job_index][3] == 0:
            node = []
            traffic = group_traffic_single_link[job_index]
            if np.sum(reverse_gpu) < job_set[job_index][4] * unit_gpu_num[job_index]:
                undeploy_job.append(job_index)
                continue
            for i in range(0, job_set[job_index][4]):
                pod_time = [max(output_oxc[j] / oxc_tor_bandwidth_rate, output_tor[j]) for j in range(0, pod)]
                rank_node = np.argmin(pod_time)
                if rank_node not in node:
                    output_oxc[rank_node] += traffic
                rank_node_gpu = min(reverse_gpu[rank_node], unit_gpu_num[job_index])
                copy_node = unit_gpu_num - rank_node_gpu
                output_tor[rank_node] += rank_node_gpu * traffic / unit_gpu_num[job_index]
                reverse_gpu[rank_node] -= rank_node_gpu
                unit_node = [rank_node]
                node.append(rank_node)
                while copy_node > 0:
                    sub_pod = [j for j in range(0, pod) if j not in unit_node]
                    sub_pod_time = [max(output_oxc[j] / oxc_tor_bandwidth_rate, output_tor[j]) for j in range(0, pod)
                                    if j not in unit_node]
                    sub_node = sub_pod[np.argmin(sub_pod_time)]
                    sub_node_gpu = min(reverse_gpu[sub_node], copy_node)
                    copy_node -= sub_node_gpu
                    output_oxc[sub_node] += sub_node_gpu * traffic / unit_gpu_num[job_index]
                    output_tor[sub_node] += sub_node_gpu * traffic / unit_gpu_num[job_index]
                    reverse_gpu[sub_node] -= sub_node_gpu
                    single_oxc_matrix[rank_node][sub_node] += 0.5 * sub_node_gpu * traffic / unit_gpu_num[job_index]
                    single_oxc_matrix[sub_node][rank_node] += 0.5 * sub_node_gpu * traffic / unit_gpu_num[job_index]
            if len(node) == 0:
                continue
            else:
                node_set = list(set(node))
                link_bool_matrix, link_job = connect_link_ring(link_bool_matrix, node_set, oxc_num)
                # link_each_job[job_index] = link_job
                output_tor += single_output_tor
                output_oxc += single_output_oxc
                single_oxc_matrix += 0.5 * traffic * link_job
                oxc_matrix += single_oxc_matrix
                # output_traffic += single_oxc_matrix
        if job_set[job_index][3] == 1:
            link_job = np.zeros([pod, pod], dtype=bool)
            traffic = group_traffic_single_link[job_index]
            ps_node = np.argmin(output_tor)
            worker_node = []
            ava_pod = []
            degree = np.sum(link_bool_matrix, axis=1)
            for i in range(0, pod):
                if i == ps_node:
                    ava_pod.append(i)
                elif link_bool_matrix[i][ps_node] == 1:
                    ava_pod.append(i)
                elif degree[i] < oxc_num and degree[ps_node] < oxc_num:
                    ava_pod.append(i)
            # output_traffic[ps_node] += job_set[job_index][4] * traffic
            output_tor[ps_node] += job_set[job_index][4] * traffic
            for i in range(0, job_set[job_index][4]):
                ava_pod = [j for j in ava_pod if unit_gpu_num[job_index] <= reverse_gpu[j]]
                if len(ava_pod) > 0:
                    output_traffic_copy = [max(output_tor[i], output_oxc[i] / oxc_tor_bandwidth_rate) for i in ava_pod]
                    min_index = ava_pod[np.argmin(output_traffic_copy)]
                    # output_traffic[min_index] -= traffic
                    reverse_gpu[job_index] -= unit_gpu_num[job_index]
                    link_job[ps_node][min_index] = 1
                    link_job[min_index][ps_node] = 1
                    if min_index not in worker_node and min_index != ps_node:
                        output_oxc[min_index] += traffic
                        output_tor[min_index] += traffic
                    else:
                        output_tor[min_index] += traffic

                    worker_node.append(min_index)
                    if link_bool_matrix[ps_node][min_index] == 0:
                        degree[ps_node] += 1
                        degree[min_index] += 1
                        if degree[ps_node] == oxc_num:
                            ava_pod = [j for j in ava_pod if link_bool_matrix[ps_node][j] == 1 or
                                       link_job[ps_node][j] == 1]
                else:
                    # output_traffic[ps_node] -= job_set[job_index][4] * traffic
                    copy_node = []
                    for j in worker_node:
                        # output_traffic[j] -= traffic
                        output_tor[j] -= traffic
                        reverse_gpu += unit_gpu_num[job_index]
                        if j not in copy_node and j != ps_node:
                            output_oxc[j] -= traffic
                        copy_node.append(j)
                    link_job = np.zeros([pod, pod], dtype=bool)
                    output_oxc[ps_node] -= job_set[job_index][4] * traffic
            # link_each_job[job_index] = link_job
            single_oxc_matrix = link_job * traffic
            oxc_matrix += single_oxc_matrix
            link_bool_matrix += link_job
        if job_set[job_index][3] == 2:
            node = []
            link_job = np.zeros([pod, pod], dtype=bool)
            for i in range(0, job_set[job_index][4]):
                enough_pod = [j for j in range(0, pod) if unit_gpu_num[job_index] <= reverse_gpu[j]]
                degree = np.sum(link_bool_matrix, axis=1)
                del_node = set()
                traffic = job_set[job_index][4] * group_traffic_single_link[job_index]
                if len(node) > 0:
                    for j in range(0, pod):
                        for k in node:
                            if link_bool_matrix[j][k] + link_job[j][k] == 0:
                                if degree[j] == oxc_num or degree[k] == oxc_num:
                                    del_node.add(j)
                enough_pod = [j for j in enough_pod if j not in del_node]
                if len(enough_pod) > 0:
                    output_traffic_enough = [max(output_tor[j], output_oxc[j] / oxc_tor_bandwidth_rate)
                                             for j in enough_pod]
                    min_index = np.argmin(output_traffic_enough)
                    # output_traffic[min_index] += traffic
                    reverse_gpu[min_index] -= unit_gpu_num[job_index]
                    for j in node:
                        link_job[min_index][j] = 1
                        link_job[j][min_index] = 1
                    node.append(min_index)
                else:
                    for j in node:
                        # output_traffic[j] -= traffic
                        reverse_gpu[j] += unit_gpu_num[job_index]
                    link_job = np.zeros([pod, pod])
                    break
            link_bool_matrix += link_job
            single_oxc_matrix = link_job * group_traffic_single_link[job_index]
            oxc_matrix += single_oxc_matrix
            for i in node:
                output_oxc[i] += (job_set[job_index][4] - 1) * group_traffic_single_link[job_index]
                output_tor[i] += (job_set[job_index][4] - 1) * group_traffic_single_link[job_index]
    return output_tor, oxc_matrix


def connect_link_ring(link_matrix, node_deploy, oxc_port):
    """

    :param oxc_port:
    :param link_matrix:
    :param node_deploy:
    :return:
    """
    rep_link = np.zeros_like(link_matrix)
    pod_num = link_matrix.shape[0]
    for i in range(0, pod_num):
        for j in range(0, pod_num):
            if link_matrix[i][j] == 1:
                if i in node_deploy and j in node_deploy:
                    rep_link[i][j] = 1
    rep_degree = np.sum(rep_link, axis=0)
    ring = []
    for i in node_deploy:
        while rep_degree[i] > 2:
            adjoin_node = [j for j in range(0, pod_num) if rep_link[i][j] == 1]
            adjoin_degree = [rep_degree[j] for j in adjoin_node]
            max_index = adjoin_node[np.argmax(adjoin_degree)]
            rep_link[i][max_index] = 0
            rep_link[max_index][i] = 0
            rep_degree[i] -= 1
            rep_degree[max_index] -= 1
    test_node = copy.deepcopy(node_deploy)
    while len(test_node) > 0:
        sub_ring = [test_node[0]]
        test_node = [i for i in test_node if i != test_node[0]]
        while 1:
            adjoin = [i for i in range(0, pod_num) if rep_link[sub_ring[-1]][i] == 1]
            if len(sub_ring) == 1:
                next_node = adjoin[0]
            else:
                adjoin = [i for i in adjoin if i != sub_ring[-1]]
                if len(adjoin) == 0:
                    break
                else:
                    if adjoin[0] == sub_ring[0]:
                        break
                    else:
                        next_node = adjoin[0]
            test_node = [i for i in test_node if i != next_node]
            sub_ring.append(next_node)
        ring += sub_ring
    link_single_matrix = np.zeros([pod_num, pod_num])
    for i in range(0, len(ring) - 1):
        link_single_matrix[ring[i]][ring[i + 1]] = 1
        link_single_matrix[ring[i + 1]][ring[i]] = 1
    link_single_matrix[ring[0]][ring[-1]] = 1
    link_single_matrix[ring[-1]][ring[0]] = 1
    degree_after = np.sum(link_single_matrix + link_matrix, axis= 0)
    if sum([1 for i in range(0, pod_num) if degree_after[i] > oxc_port]) > 0:
        link_single_matrix = np.zeros([pod_num, pod_num])
    link_matrix += link_single_matrix
    return link_matrix, link_single_matrix


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
