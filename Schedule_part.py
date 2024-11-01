import copy
import time

import numpy as np


# import random


def mountain_climb(job_set, pod_num, unit_gpu, per_oxc_port, b_tor, b_oxc, t_recon, gpu_usage, gpu_flops):
    """
    爬山法迭代最优解
    :param gpu_flops:
    :param gpu_usage:
    :param t_recon:
    :param b_oxc:
    :param b_tor:
    :param per_oxc_port:
    :param unit_gpu:
    :param pod_num:
    :param job_set: 业务六元组集合（业务索引，业务参数量，context length，通信模式， 并行度， 单个并行单元的 gpu 数目）, 通信模式分 ring, ps,
    all_to_all 三类，分别使用0， 1， 2 表示
    :return:
    """
    a = [i for i in range(0, len(job_set))]
    b = []
    list_t = []
    used_group = [a]
    topo_bench = np.zeros([pod_num, pod_num])
    for i in range(0, pod_num):
        sub_group = int(i / 8)
        for j in range(sub_group * 8, sub_group * 8):
            if j != i:
                topo_bench[i][j] += 1
                topo_bench[j][i] += 1
    single_link, sum_traffic = traffic_count(job_set)
    train_time = [train_server(job_set[i][2], job_set[i][1], gpu_flops, gpu_usage, job_set[i][4] * job_set[i][5]) for i
                  in range(0, len(job_set))]
    solution, undeploy = deploy_server(a, job_set, single_link, sum_traffic, pod_num, unit_gpu, 1, 8)
    a = [i for i in a if i not in undeploy]
    print("finish deploy")
    # print(solution, undeploy)
    traffic_push, tor_push, traffic_pull, tor_pull = traffic_topo(a, job_set, solution, pod_num, single_link)
    t_iter_push_a, topo_iter_push_a = oxc_count(traffic_push, tor_push, per_oxc_port, b_tor, b_oxc)
    t_iter_pull_a, topo_iter_pull_a = oxc_count(traffic_pull, tor_pull, per_oxc_port, b_tor, b_oxc)
    if t_iter_push_a == -1 or t_iter_pull_a == -1:
        t_iter = np.inf
    else:
        train_time_1 = [train_time[i] for i in range(0, len(train_time)) if job_set[i][3] != 1]
        t_iter = t_iter_push_a + t_iter_pull_a + max(train_time + [0]) + max(train_time_1 + [0]) + 0.2
    # print(t_iter)
    t_testbench = t_iter + 0
    list_t.append(t_testbench)
    k = 1
    flag = 0
    traffic_a = np.zeros([pod_num, pod_num])
    traffic_b = np.zeros([pod_num, pod_num])
    while 1:
        # print("iteration" + str(k))
        k += 1
        t_iter_list = np.zeros(len(a) + len(b))
        """
        if flag == len(a + b):
            return t_iter
        """
        for i in a + b:
            ai = copy.deepcopy(a)
            bi = copy.deepcopy(b)
            if i in a:
                ai = [j for j in a if j != i]
                """
                if ai in used_group:
                    flag += 1
                    continue
                else:
                    flag = 0
                """
                bi += [i]
            else:
                bi = [j for j in b if j != i]
                ai += [i]
                """
                if ai in used_group:
                    flag += 1
                    continue
                else:
                    flag = 0
                """
            traffic_push_a, tor_push_a, traffic_pull_a, tor_pull_a = (
                traffic_topo(ai, job_set, solution, pod_num, single_link))
            t_iter_push_a, topo_iter_push_a = (
                oxc_count(traffic_push_a, tor_push_a, per_oxc_port, b_tor, b_oxc))
            t_iter_pull_a, topo_iter_pull_a = (
                oxc_count(traffic_pull_a, tor_pull_a, per_oxc_port, b_tor, b_oxc))
            traffic_push_b, tor_push_b, traffic_pull_b, tor_pull_b = (
                traffic_topo(bi, job_set, solution, pod_num, single_link))
            t_iter_push_b, topo_iter_push_b = (
                oxc_count(traffic_push_b, tor_push_b, per_oxc_port, b_tor, b_oxc))
            t_iter_pull_b, topo_iter_pull_b = (
                oxc_count(traffic_pull_b, tor_pull_b, per_oxc_port, b_tor, b_oxc))
            if t_iter_push_a == -1 or t_iter_pull_a == -1 or t_iter_push_b == -1 or t_iter_pull_b == -1:
                t_iter_list[i] = np.inf
                used_group.append(ai)
                # print('A')
            else:
                t_recon_1 = adjoin_topo(topo_iter_push_a, topo_iter_push_b, per_oxc_port) * t_recon
                t_recon_2 = adjoin_topo(topo_iter_push_b, topo_iter_pull_a, per_oxc_port) * t_recon
                t_recon_3 = adjoin_topo(topo_iter_pull_a, topo_iter_pull_b, per_oxc_port) * t_recon
                t_recon_4 = adjoin_topo(topo_iter_pull_b, topo_iter_push_a, per_oxc_port) * t_recon
                train_a = max([0] + [train_time[j] for j in ai])
                train_a_agg = max([0] + [train_time[j] for j in ai if job_set[j][3] != 1])
                train_b = max([0] + [train_time[j] for j in bi])
                train_b_agg = max([0] + [train_time[j] for j in bi if job_set[j][3] != 1])
                """
                print(ai, bi)
                print(t_iter_pull_b, train_b, t_iter_push_b, train_b_agg)
                print(train_a, t_iter_push_a, train_a_agg, t_iter_pull_a)
                """
                t_iter_1 = max(t_iter_pull_b + t_recon_4, train_a)
                t_iter_2 = max(t_iter_push_a + t_recon_1, train_b)
                t_iter_3 = max(t_iter_push_b + t_recon_2, train_a_agg)
                t_iter_4 = max(t_iter_pull_a + t_recon_3, train_b_agg)
                t_iter_list[i] = t_iter_1 + t_iter_2 + t_iter_3 + t_iter_4
                used_group.append(ai)
                # print(i, t_iter_list[i])
                if t_iter_list[i] == 0:
                    print(1)
                    print(1)
        # print(t_iter)
        if min(t_iter_list) == np.inf:
            b += [a[0]]
            a = [j for j in a if j != a[0]]
        elif np.min(t_iter_list) > t_iter:
            traffic_push_a, tor_push_a, traffic_pull_a, tor_pull_a = (
                traffic_topo(a, job_set, solution, pod_num, single_link))
            traffic_push_b, tor_push_b, traffic_pull_b, tor_pull_b = (
                traffic_topo(a, job_set, solution, pod_num, single_link))
            t_push_a = time_count(topo_bench, b_oxc, traffic_push_a, pod_num, tor_push_a, b_tor)
            t_push_b = time_count(topo_bench, b_oxc, traffic_push_b, pod_num, tor_push_b, b_tor)
            t_pull_a = time_count(topo_bench, b_oxc, traffic_pull_a, pod_num, tor_pull_a, b_tor)
            t_pull_b = time_count(topo_bench, b_oxc, traffic_pull_b, pod_num, tor_pull_b, b_tor)
            train_a = max([0] + [train_time[j] for j in a])
            train_a_agg = max([0] + [train_time[j] for j in a if job_set[j][3] != 1])
            train_b = max([0] + [train_time[j] for j in b])
            train_b_agg = max([0] + [train_time[j] for j in b if job_set[j][3] != 1])
            t_iter_1 = max(t_pull_b, train_a)
            t_iter_2 = max(t_push_a, train_b)
            t_iter_3 = max(t_push_b, train_a_agg)
            t_iter_4 = max(t_pull_a, train_b_agg)
            t_iter_no_tpe = t_iter_1 + t_iter_2 + t_iter_3 + t_iter_4
            return t_iter, t_testbench - 0.2, (t_testbench - 0.2 - t_iter) / (t_testbench - 0.2), t_iter_no_tpe, (t_iter_no_tpe - t_iter) / t_iter_no_tpe
        else:
            index = np.argmin(t_iter_list)
            if index in a:
                a = [j for j in a if j != index]
                b += [index]
                used_group.append(a)
            else:
                b = [j for j in b if j != index]
                a += [index]
            t_iter = min(t_iter_list)
            flag_l = 0
            if len(list_t) > 10:
                list_t_copy = [list_t[j] for j in range(len(list_t) - 10, len(list_t))]
                for j in range(1, len(list_t_copy)):
                    list_t_copy[j] -= list_t_copy[0]
                for j in range(1, len(list_t_copy)):
                    if list_t_copy[j] != 0:
                        flag_l = 1
                if flag_l == 0:
                    traffic_push_a, tor_push_a, traffic_pull_a, tor_pull_a = (
                        traffic_topo(a, job_set, solution, pod_num, single_link))
                    traffic_push_b, tor_push_b, traffic_pull_b, tor_pull_b = (
                        traffic_topo(a, job_set, solution, pod_num, single_link))
                    traffic_all = traffic_push_a + traffic_push_b + traffic_pull_a + traffic_pull_b
                    tor_all = tor_push_a + tor_push_b + tor_pull_a + tor_pull_b
                    _, topo_no_tpe = oxc_count(traffic_all, tor_all, per_oxc_port, b_tor, b_oxc)
                    t_push_a = time_count(topo_bench, b_oxc, traffic_push_a, pod_num, tor_push_a, b_tor)
                    t_push_b = time_count(topo_bench, b_oxc, traffic_push_b, pod_num, tor_push_b, b_tor)
                    t_pull_a = time_count(topo_bench, b_oxc, traffic_pull_a, pod_num, tor_pull_a, b_tor)
                    t_pull_b = time_count(topo_bench, b_oxc, traffic_pull_b, pod_num, tor_pull_b, b_tor)
                    train_a = max([0] + [train_time[j] for j in a])
                    train_a_agg = max([0] + [train_time[j] for j in a if job_set[j][3] != 1])
                    train_b = max([0] + [train_time[j] for j in b])
                    train_b_agg = max([0] + [train_time[j] for j in b if job_set[j][3] != 1])
                    t_iter_1 = max(t_pull_b, train_a)
                    t_iter_2 = max(t_push_a, train_b)
                    t_iter_3 = max(t_push_b, train_a_agg)
                    t_iter_4 = max(t_pull_a, train_b_agg)
                    t_iter_no_tpe = t_iter_1 + t_iter_2 + t_iter_3 + t_iter_4
                    return t_iter, t_testbench - 0.2, (t_testbench - 0.2 - t_iter) / (
                                t_testbench - 0.2), t_iter_no_tpe, (t_iter_no_tpe - t_iter) / t_iter_no_tpe
            list_t.append(t_iter)


def time_count(topo, b_oxc, traffic, pod_num, tor_traffic, b_tor):
    t = 0
    for i in range(0, pod_num):
        for j in range(0, pod_num):
            if traffic[i][j] > 0:
                if traffic[i][j] / b_oxc > t:
                    t = traffic[i][j] / b_oxc
    for i in range(0, pod_num):
        if tor_traffic[i] > 0:
            if tor_traffic[i] / b_tor > t:
                t = tor_traffic[i] / b_tor
    return t


def adjoin_topo(topo_1, topo_2, oxc_port):
    topo_joint = np.zeros_like(topo_1)
    pod_num = topo_joint.shape[1]
    for i in range(0, pod_num):
        for j in range(0, pod_num):
            topo_joint[i][j] = max(topo_1[i][j] + topo_1[j][i], topo_2[i][j] + topo_2[j][i])
    use_port = np.sum(topo_joint, axis=1)
    for i in range(0, pod_num):
        if use_port[i] > oxc_port:
            return 1
    return 0


def oxc_count(oxc_traffic, tor_traffic, oxc_port, tor_bandwidth, link_bandwidth):
    pod_num = len(tor_traffic)
    oxc_topo = np.zeros([pod_num, pod_num])
    degree_pod = np.zeros(pod_num)
    t_test = np.zeros([pod_num, pod_num])
    t_tor = np.zeros(pod_num)
    for i in range(0, pod_num):
        for j in range(0, pod_num):
            if oxc_traffic[i][j] > 0:
                oxc_topo[i][j] = 1
                t_test[i][j] = oxc_traffic[i][j] / (oxc_topo[i][j] * link_bandwidth)
                degree_pod[i] += 1
                degree_pod[j] += 1
    for i in range(0, pod_num):
        t_tor[i] = tor_traffic[i] / tor_bandwidth
    degree = np.sum(oxc_topo, axis=0) + np.sum(oxc_topo, axis=1)
    # print(degree)
    reverse_degree = np.array([oxc_port - degree[i] for i in range(0, pod_num)])
    if np.count_nonzero(reverse_degree[reverse_degree < 0]) > 0:
        return -1, np.zeros([pod_num, pod_num])
    max_index_tor = np.argmax(t_tor)
    while 1:
        max_index = np.argmax(t_test)
        max_row_oxc = int(max_index / pod_num)
        max_col_oxc = int(max_index % pod_num)
        if t_test[max_row_oxc][max_col_oxc] <= t_tor[max_index_tor]:
            break
        else:
            if degree_pod[max_row_oxc] < oxc_port and degree_pod[max_col_oxc] < oxc_port:
                degree_pod[max_row_oxc] += 1
                degree_pod[max_col_oxc] += 1
                oxc_topo[max_row_oxc][max_col_oxc] += 1
                t_test[max_row_oxc][max_col_oxc] = (oxc_traffic[max_row_oxc][max_col_oxc] /
                                                    (oxc_topo[max_row_oxc][max_col_oxc] * link_bandwidth))
            else:
                break
    t_min = max(t_tor[max_index_tor], np.max(t_test))
    return t_min, oxc_topo


def traffic_topo(group, job_set, local_solution, pod_num, traffic_single_link):
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
    traffic_matrix_push = np.zeros([pod_num, pod_num])
    tor_traffic_push = np.zeros(pod_num)
    traffic_matrix_pull = np.zeros([pod_num, pod_num])
    tor_traffic_pull = np.zeros(pod_num)
    # link_bool_a_1 = np.zeros([pod_num, pod_num])
    for n in range(0, 2):
        link_bool_a_1 = np.zeros([pod_num, pod_num])
        for i in group:
            if len(local_solution[i]) == 0:
                continue
            single_oxc_traffic = np.zeros([pod_num, pod_num])
            if job_set[i][3] == 0:
                ring_a.append(i)
            elif job_set[i][3] == 1:
                local_job = local_solution[i]
                ps_node = local_job[0][0]
                for j in range(1, job_set[i][4] + 1):
                    unit_job = local_job[j]
                    rank_node, rank_node_gpu = unit_job[0][0], unit_job[0][1]
                    tor_traffic_push[rank_node] += rank_node_gpu * traffic_single_link[i] / job_set[i][5]
                    if rank_node != ps_node:
                        if n == 0:
                            single_oxc_traffic[rank_node][ps_node] += traffic_single_link[i]
                        else:
                            single_oxc_traffic[ps_node][rank_node] += traffic_single_link[i]
                        # link_bool_a_1[rank_node][ps_node] = 1
                    for k in range(1, len(unit_job)):
                        sub_node, sub_node_gpu = unit_job[k][0], unit_job[k][1]
                        # link_bool_a_1[rank_node][sub_node] = 1
                        single_oxc_traffic[rank_node][sub_node] += (
                                sub_node_gpu * traffic_single_link[i] / job_set[i][5])
                        tor_traffic_push[sub_node] += sub_node_gpu * traffic_single_link[i] / job_set[i][5]
            elif job_set[i][3] == 2:
                local_job = local_solution[i]
                for j in range(0, job_set[i][4]):
                    unit_job = local_job[j]
                    for k in range(0, len(unit_job)):
                        node, node_gpu = unit_job[k][0], unit_job[k][1]
                        tor_traffic_push[node] += (
                                node_gpu * traffic_single_link[i] * (job_set[i][4] - 1) / job_set[i][5])
                        if k > 0:
                            single_oxc_traffic[node][unit_job[0][0]] += (
                                    node_gpu * traffic_single_link[i] * (job_set[i][4] - 1) / job_set[i][5])
                for p in range(0, job_set[i][4]):
                    for q in range(0, job_set[i][4]):
                        if p != q:
                            if local_job[p][0][0] != local_job[q][0][0]:
                                single_oxc_traffic[local_job[p][0][0]][local_job[q][0][0]] += traffic_single_link[i]
            if n == 0:
                traffic_matrix_push += single_oxc_traffic
            else:
                traffic_matrix_pull += single_oxc_traffic
            """
            for i in range(0, pod_num):
                for j in range(0, pod_num):
                    if traffic_matrix[i][j] > 0:
                        link_bool_a_1[i][j] = 1
            """
        if n == 0:
            link_bool_a_1 = (traffic_matrix_push > 0).astype(int)
        else:
            link_bool_a_1 = (traffic_matrix_pull > 0).astype(int)
        for i in ring_a:
            ring_oxc_traffic, ring_tor_traffic = (
                connect_link_ring(link_bool_a_1, traffic_single_link[i], local_solution[i], job_set[i]))
            link_bool_a_1 = (ring_oxc_traffic > 0).astype(int)
            if n == 0:
                traffic_matrix_push += ring_oxc_traffic
                tor_traffic_push += ring_tor_traffic
            else:
                traffic_matrix_pull += ring_oxc_traffic
                tor_traffic_pull += ring_tor_traffic
    return traffic_matrix_push, tor_traffic_push, traffic_matrix_pull, tor_traffic_pull


def deploy_server(group, job_set, group_traffic_single_link, group_traffic_size, pod, pod_gpu, oxc_tor_bandwidth_rate,
                  seg_pod_num):
    """
    各组并行单元的部署方案生成，产生拓扑，注意连接选择半双工，不使用INC。部署方案改为：将所有 pod 分小组，每个业务只放进一个小组
    :return:
    """
    local_solution = np.empty(len(group), dtype=object)
    reverse_gpu = np.array([pod_gpu for _ in range(0, pod)])
    # output_traffic = np.zeros(pod)
    group_traffic = [group_traffic_size[i] for i in group]
    group_traffic_copy = copy.deepcopy(group_traffic)
    undeploy_job = []
    seg_num = int(pod / seg_pod_num)
    seg_traffic = np.zeros(seg_num)
    seg_pod_reserve = np.array([seg_pod_num * pod_gpu for _ in range(0, seg_num)])
    # link_bool_matrix = np.zeros([pod, pod])
    # link_each_job = np.empty(len(job_set), dtype=None)
    output_oxc = np.zeros(pod)
    output_tor = np.zeros(pod)
    # oxc_matrix = np.zeros([pod, pod])
    unit_gpu_num = np.zeros(len(group))
    for i in range(0, len(group)):
        unit_gpu_num[i] = job_set[i][5]
    while np.max(group_traffic_copy) > 0:
        job_index = group[np.argmax(group_traffic_copy)]
        seg_traffic_copy = copy.deepcopy(seg_traffic)
        # print(seg_pod_reserve)
        for i in range(0, seg_num):
            if job_set[job_index][3] == 1:
                if seg_pod_reserve[i] < job_set[job_index][4] * job_set[job_index][5] + 1:
                    seg_traffic_copy[i] = np.inf
            else:
                if seg_pod_reserve[i] < job_set[job_index][4] * job_set[job_index][5]:
                    seg_traffic_copy[i] = np.inf
        if min(seg_traffic_copy) == np.inf:
            undeploy_job.append(job_index)
            local_solution[job_index] = []
            group_traffic_copy[np.argmax(group_traffic_copy)] = -1
            continue
        sub_pod_index = int(np.argmin(seg_traffic_copy))
        # print(job_index, seg_traffic_copy, sub_pod_index, np.argmin(seg_traffic_copy))
        group_traffic_copy[np.argmax(group_traffic_copy)] = -1
        # single_output_oxc = np.zeros(pod)
        # single_output_tor = np.zeros(pod)
        # if np.sum(reverse_gpu) < job_set[job_index][4] * unit_gpu_num[job_index]:
        # single_oxc_matrix = np.zeros([pod, pod])
        if job_set[job_index][3] == 0:
            local_job = []
            node = []
            traffic = group_traffic_single_link[job_index]
            for i in range(0, job_set[job_index][4]):
                unit_local = []
                output_traffic_copy = [max(output_oxc[j] / oxc_tor_bandwidth_rate, output_tor[j]) if reverse_gpu[j] > 0
                                       else np.inf for j in
                                       range(int(sub_pod_index * seg_pod_num), int((sub_pod_index + 1) * seg_pod_num))]
                for j in node:
                    if reverse_gpu[j] > 0:
                        output_traffic_copy[int(j - sub_pod_index * seg_pod_num)] = output_tor[j]
                rank_node = int(np.argmin(output_traffic_copy) + sub_pod_index * seg_pod_num)
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
                                   else np.inf for j in
                                   range(int(sub_pod_index * seg_pod_num), int((sub_pod_index + 1) * seg_pod_num))]
                    sub_node = int(np.argmin(sub_traffic) + sub_pod_index * seg_pod_num)
                    sub_node_gpu = min(reverse_gpu[sub_node], copy_node)
                    # print(sub_node, sub_node_gpu, copy_node, sub_traffic, reverse_gpu[int(sub_pod_index * seg_pod_num): int((sub_pod_index + 1) * seg_pod_num)])
                    copy_node -= sub_node_gpu
                    output_oxc[sub_node] += sub_node_gpu * traffic / unit_gpu_num[job_index]
                    output_tor[sub_node] += sub_node_gpu * traffic / unit_gpu_num[job_index]
                    seg_traffic[sub_pod_index] += sub_node_gpu * traffic / unit_gpu_num[job_index]
                    reverse_gpu[sub_node] -= sub_node_gpu
                    unit_local.append((sub_node, sub_node_gpu))
                local_job.append(unit_local)
                # print(job_set[job_index][4] * job_set[job_index][5])
            seg_pod_reserve[sub_pod_index] -= job_set[job_index][4] * job_set[job_index][5]
            seg_traffic[sub_pod_index] += group_traffic_size[job_index]
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
                                                                                                 > 0 else np.inf for i
                                   in
                                   range(int(sub_pod_index * seg_pod_num), int((sub_pod_index + 1) * seg_pod_num))]
            traffic = group_traffic_single_link[job_index]
            ps_node = int(sub_pod_index * seg_pod_num + np.argmin(output_traffic_copy))
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
            # output_oxc[ps_node] += job_set[job_index][4] * traffic
            reverse_gpu[ps_node] -= 1
            for i in range(0, job_set[job_index][4]):
                unit_local = []
                # print('unit' + str(i + 1))
                # ava_pod = [j for j in ava_pod if unit_gpu_num[job_index] <= reverse_gpu[j]]
                output_traffic_copy = np.array([max(output_tor[j], output_oxc[j] / oxc_tor_bandwidth_rate) if
                                                reverse_gpu[j] > 0 else np.inf for j in
                                                range(int(sub_pod_index * seg_pod_num),
                                                      int((sub_pod_index + 1) * seg_pod_num))])
                if reverse_gpu[ps_node] > 0:
                    output_traffic_copy[int(ps_node - seg_pod_num * sub_pod_index)] = output_tor[ps_node]
                rank_node = int(np.argmin(output_traffic_copy) + sub_pod_index * seg_pod_num)
                # output_traffic[min_index] -= traffic
                # link_job[ps_node][min_index] = 1
                # link_job[min_index][ps_node] = 1
                worker_node.append(rank_node)
                rank_node_gpu = min(unit_gpu_num[job_index], reverse_gpu[rank_node])
                reverse_gpu[rank_node] -= rank_node_gpu
                unit_local.append((rank_node, rank_node_gpu))
                copy_node = unit_gpu_num[job_index] - rank_node_gpu
                if rank_node != ps_node:
                    output_tor[rank_node] += traffic * rank_node_gpu / unit_gpu_num[job_index]
                    output_oxc[rank_node] += traffic
                    output_oxc[ps_node] += traffic
                else:
                    output_tor[rank_node] += traffic * rank_node_gpu / unit_gpu_num[job_index]
                while copy_node > 0:
                    sub_traffic = [max(output_tor[j], output_oxc[j] / oxc_tor_bandwidth_rate) if reverse_gpu[j]
                                                                                                 > 0 else np.inf for j
                                   in
                                   range(int(sub_pod_index * seg_pod_num), int((sub_pod_index + 1) * seg_pod_num))]
                    sub_node = int(np.argmin(sub_traffic) + sub_pod_index * seg_pod_num)
                    # print('sub_node')
                    sub_node_gpu = min(reverse_gpu[sub_node], copy_node)
                    # print(sub_node, sub_node_gpu, copy_node, sub_traffic)
                    copy_node -= sub_node_gpu
                    reverse_gpu[sub_node] -= sub_node_gpu
                    output_oxc[rank_node] += traffic * sub_node_gpu / unit_gpu_num[job_index]
                    output_tor[rank_node] += traffic * sub_node_gpu / unit_gpu_num[job_index]
                    output_oxc[sub_node] += traffic * sub_node_gpu / unit_gpu_num[job_index]
                    output_tor[sub_node] += traffic * sub_node_gpu / unit_gpu_num[job_index]
                    seg_traffic[sub_pod_index] += traffic * sub_node_gpu / unit_gpu_num[job_index]
                    unit_local.append((sub_node, sub_node_gpu))
                local_job.append(unit_local)
            local_solution[job_index] = local_job
            seg_traffic[sub_pod_index] += group_traffic_size[job_index]
            seg_pod_reserve[sub_pod_index] -= job_set[job_index][4] * unit_gpu_num[job_index] + 1
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
            # link_job = np.zeros([pod, pod], dtype=bool)
            local_job = []
            reverse_gpu_ep = [reverse_gpu[j] for j in
                              range(int(sub_pod_index * seg_pod_num), int((sub_pod_index + 1) * seg_pod_num))]
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
                rank_node = int(np.argmax(reverse_gpu_ep) + sub_pod_index * seg_pod_num)
                rank_node_gpu = min(reverse_gpu[rank_node], unit_gpu_num[job_index])
                # output_traffic[min_index] += traffic
                reverse_gpu[rank_node] -= rank_node_gpu
                reverse_gpu_ep[int(rank_node - sub_pod_index * seg_pod_num)] -= rank_node_gpu
                unit_local.append((rank_node, rank_node_gpu))
                copy_node = unit_gpu_num[job_index] - rank_node_gpu
                node.append(rank_node)
                k = 0
                while copy_node > 0:
                    if k > 9:
                        break
                    k += 1
                    sub_node = int(np.argmax(reverse_gpu_ep) + sub_pod_index * seg_pod_num)
                    sub_node_gpu = min(copy_node, reverse_gpu[sub_node])
                    reverse_gpu_ep[int(sub_node - sub_pod_index * seg_pod_num)] -= sub_node_gpu
                    reverse_gpu[sub_node] -= sub_node_gpu
                    seg_traffic[sub_pod_index] += (group_traffic_single_link[job_index] * (job_set[job_index][4] - 1) *
                                                   sub_node_gpu / unit_gpu_num[job_index])
                    # print(sub_node, sub_node_gpu, copy_node)
                    unit_local.append((sub_node, sub_node_gpu))
                    copy_node -= sub_node_gpu
                local_job.append(unit_local)
            if len(set(node)) > 1:
                for i in set(node):
                    output_oxc[i] += group_traffic_single_link[job_index] * (len(set(node)) - 1)
            for i in range(0, len(local_job)):
                for j in range(0, len(local_job[i])):
                    output_tor[local_job[i][j][0]] += \
                        (group_traffic_single_link[job_index] * (job_set[job_index][4] - 1) * local_job[i][j][1] /
                         unit_gpu_num[job_index])
            local_solution[job_index] = local_job
            seg_traffic[sub_pod_index] += group_traffic_size[job_index]
            seg_pod_reserve[sub_pod_index] -= job_set[job_index][4] * job_set[job_index][5]
    return local_solution, undeploy_job


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
            adjoin_degree = np.array([out_degree[j] for j in adjoin_node])
            max_index = adjoin_node[np.argmax(adjoin_degree)]
            rep_link[max_index][i] = 0
            in_degree[i] -= 1
            out_degree[max_index] -= 1
        while out_degree[i] > 1:
            adjoin_node = [j for j in range(0, pod_num) if rep_link[i][j] == 1]
            adjoin_degree = np.array([in_degree[j] for j in adjoin_node])
            max_index = adjoin_node[np.argmax(adjoin_degree)]
            rep_link[i][max_index] = 0
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
                test_node = [i for i in test_node if i != next_node]
                sub_ring.append(next_node)
        ring += sub_ring
    oxc_traffic = np.zeros([pod_num, pod_num])
    tor_traffic = np.zeros(pod_num)
    for i in range(0, len(ring) - 1):
        oxc_traffic[ring[i]][ring[i + 1]] = single_traffic
    oxc_traffic[-1][0] = single_traffic
    for i in range(0, len(local_job)):
        unit_job = local_job[i]
        for j in range(0, len(unit_job)):
            node, node_gpu = unit_job[j][0], unit_job[j][1]
            tor_traffic[node] = node_gpu * single_traffic / job_info[5]
            if j >= 1:
                oxc_traffic[node][unit_job[0][0]] = node_gpu * single_traffic / job_info[5]
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


def train_server(context, model_parameter, flops, usage, gpu_num):
    """
    业务单 GPU 训练用时
    :param gpu_num:
    :param context: 每次迭代的文本长度
    :param model_parameter: 业务参数量
    :param flops: GPU 峰值 flops
    :param usage: GPU 利用率
    :return:
    """
    return 8 * context * model_parameter / (flops * usage * gpu_num)


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
            traffic.append((job_set[i][2] * 4) * 1e-6 / (ep * ep))
            sum_output_traffic.append((ep - 1) * job_set[i][2] * 1e-6 / ep)
    return traffic, sum_output_traffic


def generate_job(job_num):
    job_set1 = []
    job_set2 = []
    job_set3 = []
    job_set4 = []
    job_set5 = []
    for i in range(0, job_num):
        job_size = np.random.randint(0, 4)
        para = [2.46, 6.42, 21.2, 177.6]
        job_para = para[job_size]
        # context = 8
        # job_para = np.random.choice(np.array([1, 3, 8, 70]), p=np.array([0.25, 0.25, 0.25]))
        context = np.random.choice(np.array([8, 16, 32, 64, 128]), p=np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        job_type = np.random.choice(np.array([0, 1, 2]), p=np.array([1 / 3, 1 / 3, 1 / 3]))
        parallel = np.random.choice(np.array([4, 6, 8, 16]), p=np.array([0.25, 0.25, 0.25, 0.25]))
        # token_per_batch =
        # np.random.choice(np.array([128, 256, 512, 1024, 2048]), p=np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        unit_gpu = np.random.randint(100, 200)
        if job_size == 3:
            unit_gpu = np.random.randint(150, 200)
            parallel = 16
        elif job_size == 2:
            parallel = np.random.choice(np.array([8, 16]), p=np.array([0.5, 0.5]))
        elif job_size == 0:
            unit_gpu = np.random.randint(50, 100)
        job_default = [i, job_para, 8, job_type, parallel, unit_gpu]
        job_context_2 = [i, job_para, 16, job_type, parallel, unit_gpu]
        job_context_3 = [i, job_para, 32, job_type, parallel, unit_gpu]
        job_context_4 = [i, job_para, 64, job_type, parallel, unit_gpu]
        job_context_5 = [i, job_para, 128, job_type, parallel, unit_gpu]

        job_set1.append(job_default)
        job_set2.append(job_context_2)
        job_set3.append(job_context_3)
        job_set4.append(job_context_4)
        job_set5.append(job_context_5)
    return job_set1, job_set2, job_set3, job_set4, job_set5


for k in range(0, 6):
    info = generate_job(100)
    job = info[0]
    job1 = job[0:100]
    start = time.time()
    t1 = mountain_climb(job1, pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=3200, b_oxc=200, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    job1 = job[0:80]
    start = time.time()
    t1 = mountain_climb(job1, pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=3200, b_oxc=200, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    job1 = job[0:60]
    start = time.time()
    t1 = mountain_climb(job1, pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=3200, b_oxc=200, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    job1 = job[0:40]
    start = time.time()
    t1 = mountain_climb(job1, pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=3200, b_oxc=200, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    job1 = job[0:20]
    start = time.time()
    t1 = mountain_climb(job1, pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=3200, b_oxc=200, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    print("------job num----------")
    start = time.time()
    t1 = mountain_climb(info[1], pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=3200, b_oxc=200, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    t1 = mountain_climb(info[2], pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=3200, b_oxc=200, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    t1 = mountain_climb(info[3], pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=3200, b_oxc=200, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    t1 = mountain_climb(info[4], pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=3200, b_oxc=200, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    print("-------context_length------")
    start = time.time()
    t1 = mountain_climb(job, pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=800, b_oxc=50, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    start = time.time()
    t1 = mountain_climb(job, pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=1600, b_oxc=100, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    start = time.time()
    t1 = mountain_climb(job, pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=2400, b_oxc=150, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    start = time.time()
    t1 = mountain_climb(job, pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=4000, b_oxc=250, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
"""
job1 = job[0]
job2 = job[1]
job3 = job[2]
job4 = job[3]
for i in range(1, 3):
    start = time.time()
    t1 = mountain_climb(job1, pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=800 * i * 2, b_oxc=50 * i * 2, t_recon=0.1, gpu_usage=0.4,
                       gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    start = time.time()
    t1 = mountain_climb(job2, pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=800 * i * 2, b_oxc=50 * i * 2, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    start = time.time()
    t1 = mountain_climb(job3, pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=800 * i, b_oxc=50 * i * 2, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    start = time.time()
    t1 = mountain_climb(job4, pod_num=256, unit_gpu=2048, per_oxc_port=16, b_tor=800 * i, b_oxc=50 * i * 2, t_recon=0.1,
                        gpu_usage=0.4,
                        gpu_flops=26)
    end = time.time()
    print(t1, end - start)
    print('--------------------------------------')
"""
