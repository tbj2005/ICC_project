import copy
import numpy as np

GPU_usage = 0.8
GPU_flops = 9.7


def mountain_climb(job_set, pod_num, unit_gpu, per_oxc_port):
    """
    爬山法迭代最优解
    :param per_oxc_port:
    :param unit_gpu:
    :param pod_num:
    :param job_set: 业务六元组集合（业务索引，业务参数量，batch size，通信模式， 并行度， 单条数据 token 量， 单个并行单元的 gpu 数目）, 通信模式分 ring, ps,
    all_to_all 三类，分别使用0， 1， 2 表示
    :return:
    """
    a = [job_set[i][0] for i in range(0, len(job_set))]
    b = []
    single_link, sum_traffic = traffic_count(job_set)
    solution = deploy_server(a, job_set, single_link, sum_traffic, pod_num, unit_gpu, per_oxc_port
                             , 1)


def topology_deploy_push(group, job_set, local_solution, pod_num, traffic_single_link):
    """
    计算每组的 OXC 拓扑，并求解对应的单迭代时长
    :param traffic_single_link:
    :param pod_num:
    :param local_solution:
    :param group: a 组业务索引
    :param job_set: 业务六元组集合
    :return:
    """
    ring_a = []
    traffic_matrix = np.zeros([pod_num, pod_num])
    for i in group:
        single_oxc_traffic = np.zeros([pod_num, pod_num])
        tor_traffic = np.zeros(pod_num)
        if job_set[i][3] == 0:
            ring_a.append(i)
        elif job_set[i][3] == 1:
            local_job = local_solution[i]
            ps_node = local_job[0][0]
            for j in range(1, job_set[i][4] + 1):
                unit_job = local_job[j]
                rank_node, rank_node_gpu = unit_job[0][0], unit_job[0][1]
                tor_traffic[rank_node] += rank_node_gpu * traffic_single_link[i] / job_set[i][6]
                single_oxc_traffic[rank_node][ps_node] += traffic_single_link
                # link_bool_a_1[rank_node][ps_node] = 1
                for k in range(1, len(unit_job)):
                    sub_node, sub_node_gpu = unit_job[k][0], unit_job[k][1]
                    # link_bool_a_1[rank_node][sub_node] = 1
                    single_oxc_traffic[rank_node][sub_node] += (
                            sub_node_gpu * traffic_single_link[i] / job_set[i][6])
                    tor_traffic[sub_node] += sub_node_gpu * traffic_single_link[i] / job_set[i][6]
        elif job_set[i][3] == 2:
            local_job = local_solution[i]
            for j in range(0, job_set[i][4]):
                unit_job = local_job[j]
                for k in range(0, len(unit_job)):
                    node, node_gpu = unit_job[0][0], unit_job[0][1]
                    tor_traffic[node] += node_gpu * traffic_single_link[i] * job_set[i][4] / job_set[i][6]
            for p in range(0, job_set[i][4]):
                for q in range(0, job_set[i][4]):
                    if p != q:
                        if local_job[p][0][0] != local_job[q][0][0]:
                            single_oxc_traffic[local_job[p][0][0]][local_job[q][0][0]] += traffic_single_link
        traffic_matrix += traffic_single_link
    link_bool_a_1 = traffic_matrix != 0
    for i in ring_a:
        connect_link_ring(link_bool_a_1, traffic_single_link[i], local_solution[i], job_set[i])


def deploy_server(group, job_set, group_traffic_single_link, group_traffic_size, pod, pod_gpu
                  , oxc_num, oxc_tor_bandwidth_rate):
    """
    各组并行单元的部署方案生成，产生拓扑，注意连接选择半双工，不使用INC
    :return:
    """
    local_solution = np.zeros(len(group), dtype=None)
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
    unit_gpu_num = np.zeros(len(group))
    for i in range(0, len(group)):
        unit_gpu_num[i] = job_set[i][6]
    while np.max(group_traffic_copy) < 0:
        job_index = group[np.argmax(group_traffic_copy)]
        group_traffic_copy[np.argmax(group_traffic_copy)] = -1
        single_output_oxc = np.zeros(pod)
        single_output_tor = np.zeros(pod)
        if np.sum(reverse_gpu) < job_set[job_index][4] * unit_gpu_num[job_index]:
            undeploy_job.append(job_index)
            continue
        # single_oxc_matrix = np.zeros([pod, pod])
        if job_set[job_index][3] == 0:
            local_job = []
            node = []
            traffic = group_traffic_single_link[job_index]
            for i in range(0, job_set[job_index][4]):
                unit_local = []
                output_traffic_copy = [max(output_oxc[j] / oxc_tor_bandwidth_rate, output_tor[j]) if reverse_gpu[j] > 0
                                       else np.inf for j in range(0, pod)]
                for j in node:
                    if reverse_gpu[j] > 0:
                        output_traffic_copy[j] = output_tor[j]
                rank_node = np.argmin(output_traffic_copy)
                if rank_node not in node:
                    output_oxc[rank_node] += traffic
                rank_node_gpu = min(reverse_gpu[rank_node], unit_gpu_num[job_index])
                copy_node = unit_gpu_num[job_index] - rank_node_gpu
                output_tor[rank_node] += traffic * rank_node_gpu / unit_gpu_num[job_index]
                reverse_gpu[rank_node] -= rank_node_gpu
                # unit_node = [rank_node]
                node.append(rank_node)
                unit_local.append((rank_node, rank_node_gpu))
                while copy_node > 0:
                    # sub_pod = [j for j in range(0, pod) if j not in unit_node]
                    sub_traffic = [max(output_oxc[j] / oxc_tor_bandwidth_rate, output_tor[j]) if reverse_gpu[j] > 0
                                   else np.inf for j in range(0, pod)]
                    sub_node = np.argmin(sub_traffic)
                    sub_node_gpu = min(reverse_gpu[sub_node], copy_node)
                    copy_node -= sub_node_gpu
                    output_oxc[sub_node] += sub_node_gpu * traffic / unit_gpu_num[job_index]
                    output_tor[sub_node] += sub_node_gpu * traffic / unit_gpu_num[job_index]
                    reverse_gpu[sub_node] -= sub_node_gpu
                    unit_local.append((sub_node, sub_node_gpu))
                local_job.append(unit_local)
            if len(node) == 0:
                continue
            else:
                # node_set = list(set(node))
                # link_bool_matrix, link_job = connect_link_ring(link_bool_matrix, node_set, oxc_num)
                # link_each_job[job_index] = link_job
                # output_tor += single_output_tor
                # output_oxc += single_output_oxc
                local_solution[job_index] = local_job
                # single_oxc_matrix += 0.5 * traffic * link_job
                # oxc_matrix += single_oxc_matrix
                # output_traffic += single_oxc_matrix
        if job_set[job_index][3] == 1:
            local_job = []
            # link_job = np.zeros([pod, pod], dtype=bool)
            output_traffic_copy = [max(output_tor[i], output_oxc[i] / oxc_tor_bandwidth_rate) if reverse_gpu[i]
                                   > 0 else np.inf for i in range(0, pod)]
            traffic = group_traffic_single_link[job_index]
            ps_node = np.argmin(output_traffic_copy)
            local_job.append((ps_node, 1))
            worker_node = []
            # ava_pod = []
            # degree = np.sum(link_bool_matrix, axis=1)
            """
            for i in range(0, pod):
                if i == ps_node:
                    ava_pod.append(i)
                elif link_bool_matrix[i][ps_node] == 1:
                    ava_pod.append(i)
                elif degree[i] < oxc_num and degree[ps_node] < oxc_num:
                    ava_pod.append(i)
            """
            # output_traffic[ps_node] += job_set[job_index][4] * traffic
            output_tor[ps_node] += job_set[job_index][4] * traffic
            output_oxc[ps_node] += job_set[job_index][4] * traffic
            reverse_gpu[ps_node] -= 1
            for i in range(0, job_set[job_index][4]):
                unit_local = []
                # ava_pod = [j for j in ava_pod if unit_gpu_num[job_index] <= reverse_gpu[j]]
                output_traffic_copy = [max(output_tor[i], output_oxc[i] / oxc_tor_bandwidth_rate) if reverse_gpu[i]
                                       > 0 else np.inf for i in range(0, pod)]
                if reverse_gpu[ps_node] > 0:
                    output_traffic_copy[ps_node] = output_tor
                rank_node = np.argmin(output_traffic_copy)
                # output_traffic[min_index] -= traffic
                # link_job[ps_node][min_index] = 1
                # link_job[min_index][ps_node] = 1
                worker_node.append(rank_node)
                rank_node_gpu = min(unit_gpu_num[job_index], reverse_gpu[rank_node])
                reverse_gpu[job_index] -= rank_node_gpu
                unit_local.append((rank_node, rank_node_gpu))
                copy_node = unit_gpu_num[job_index] - rank_node_gpu
                if rank_node != ps_node:
                    output_tor[rank_node] += traffic * rank_node_gpu / unit_gpu_num[job_index]
                    output_oxc[rank_node] += traffic
                else:
                    output_tor[rank_node] += traffic * rank_node_gpu / unit_gpu_num[job_index]
                while copy_node > 0:
                    sub_traffic = [max(output_tor[i], output_oxc[i] / oxc_tor_bandwidth_rate) if reverse_gpu[i]
                                           > 0 else np.inf for i in range(0, pod)]
                    sub_node = np.argmin(sub_traffic)
                    sub_node_gpu = min(reverse_gpu[sub_node], copy_node)
                    copy_node -= sub_node_gpu
                    reverse_gpu[sub_node] -= sub_node_gpu
                    output_oxc[rank_node] += traffic * sub_node_gpu / unit_gpu_num[job_index]
                    output_tor[rank_node] += traffic * sub_node_gpu / unit_gpu_num[job_index]
                    unit_local.append((sub_node, sub_node_gpu))
                local_job.append(unit_local)
            local_solution[job_index] = local_job
            """
            if link_bool_matrix[ps_node][min_index] == 0:
                degree[ps_node] += 1
                degree[min_index] += 1
                if degree[ps_node] == oxc_num:
                    ava_pod = [j for j in ava_pod if link_bool_matrix[ps_node][j] == 1 or
                               link_job[ps_node][j] == 1]
            """
            # link_each_job[job_index] = link_job
            # single_oxc_matrix = link_job * traffic
            # oxc_matrix += single_oxc_matrix
            # link_bool_matrix += link_job

        if job_set[job_index][3] == 2:
            node = []
            link_job = np.zeros([pod, pod], dtype=bool)
            local_job = []
            for i in range(0, job_set[job_index][4]):
                """
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
                """
                unit_local = []
                rank_node = np.argmin(reverse_gpu)
                rank_node_gpu = min(reverse_gpu[rank_node], unit_gpu_num[job_index])
                # output_traffic[min_index] += traffic
                reverse_gpu[rank_node] -= rank_node_gpu
                unit_local.append((rank_node, rank_node_gpu))
                copy_node = unit_gpu_num[job_index] - rank_node_gpu
                node.append(rank_node)
                while copy_node > 0:
                    sub_node = np.argmin(reverse_gpu)
                    sub_node_gpu = min(copy_node, reverse_gpu[sub_node])
                    reverse_gpu[sub_node] -= sub_node_gpu
                    unit_local.append((sub_node, sub_node_gpu))
                local_job.append(unit_local)
            for i in set(node):
                output_oxc[i] += group_traffic_single_link * (len(set(node)) - 1)
            for i in range(0, len(local_job)):
                for j in range(0, len(local_job[i])):
                    output_tor[local_job[i][j][0]] += (group_traffic_single_link * (job_set[job_index][4] - 1) *
                                                       local_job[i][j][1] / unit_gpu_num[job_index])
            local_solution[job_index] = local_job
    return local_solution


def connect_link_ring(link_matrix, single_traffic, local_job, job_info):
    """

    :param job_info:
    :param local_job:
    :param single_traffic:
    :param link_matrix:
    :return:
    """
    rep_link = np.zeros_like(link_matrix)
    pod_num = link_matrix.shape[0]
    node_deploy = []
    for i in range(0, len(local_job)):
        node_deploy.append(local_job[i][0][0])
    node_deploy = list(set(node_deploy))
    for i in range(0, pod_num):
        for j in range(0, pod_num):
            if link_matrix[i][j] == 1:
                if i in node_deploy and j in node_deploy:
                    rep_link[i][j] = 1
    out_degree = np.sum(rep_link, axis=1)
    in_degree = np.sum(rep_link, axis=0)
    ring = []
    for i in node_deploy:
        while in_degree[i] > 1:
            adjoin_node = [j for j in range(0, pod_num) if rep_link[j][i] == 1]
            adjoin_degree = [out_degree[j] for j in adjoin_node]
            max_index = adjoin_node[np.argmax(adjoin_degree)]
            rep_link[max_index][i] = 0
            in_degree[i] -= 1
            out_degree[max_index] -= 1
        while out_degree[i] > 1:
            adjoin_node = [j for j in range(0, pod_num) if rep_link[i][j] == 1]
            adjoin_degree = [in_degree[j] for j in adjoin_node]
            max_index = adjoin_node[np.argmax(adjoin_degree)]
            rep_link[max_index][i] = 0
            out_degree[i] -= 1
            in_degree[max_index] -= 1
    test_node = copy.deepcopy(node_deploy)
    while len(test_node) > 0:
        sub_ring = [test_node[0]]
        test_node = [i for i in test_node if i != test_node[0]]
        while 1:
            adjoin = [i for i in range(0, pod_num) if rep_link[sub_ring[-1]][i] == 1]
            if len(adjoin) == 0:
                break
            else:
                next_node = adjoin[0]
                if next_node == sub_ring[0]:
                    break
                sub_ring.append(next_node)
        ring += sub_ring
    oxc_traffic = np.zeros([pod_num, pod_num])
    tor_traffic = np.zeros(pod_num)
    for i in range(0, len(ring) - 1):
        oxc_traffic[i][i + 1] = single_traffic
    oxc_traffic[-1][0] = single_traffic
    for i in range(0, len(local_job)):
        unit_job = local_job[i]
        for j in range(0, len(unit_job)):
            node, node_gpu = unit_job[j][0], unit_job[j][1]
            tor_traffic[node] = node_gpu * single_traffic / job_info[6]
            if j >= 1:
                oxc_traffic[node][unit_job[0][0]] = node_gpu * single_traffic / job_info[6]
    """
    degree_after = np.sum(link_single_matrix + link_matrix, axis= 0)
    if sum([1 for i in range(0, pod_num) if degree_after[i] > oxc_port]) > 0:
        link_single_matrix = np.zeros([pod_num, pod_num])
    link_matrix += link_single_matrix
    """
    return oxc_traffic, tor_traffic


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
