import copy
import random
import time
from openpyxl import Workbook
import numpy as np


workbook = Workbook()
sheet = workbook.active
sheet.title = "evaluation"
data = [["TPE-PJS-heu", "TPE-PJS-HC", "TPE-JS", "TPE-heu", "PJS", "t1", "t2", "t3", "t4", "t5"]]
for row in data:
    sheet.append(row)


def init_group_old(job_set, sum_traffic, pod_num, solution):
    fa = 0
    fb = 0
    a_init = []
    b_init = []
    ps_index = np.array([job_set[i][0] for i in range(0, len(job_set)) if job_set[i][3] == 1])
    rar_index = np.array([job_set[i][0] for i in range(0, len(job_set)) if job_set[i][3] == 0])
    ep_index = np.array([job_set[i][0] for i in range(0, len(job_set)) if job_set[i][3] == 2])
    ps_size = np.array([sum_traffic[i] for i in range(0, len(job_set)) if job_set[i][3] == 1])
    rar_size = np.array([sum_traffic[i] for i in range(0, len(job_set)) if job_set[i][3] == 0])
    ep_num = np.array([job_set[i][4] for i in range(0, len(job_set)) if job_set[i][3] == 2])
    tba = np.zeros([pod_num, pod_num], dtype=bool)
    tbb = np.zeros([pod_num, pod_num], dtype=bool)
    for i in range(0, len(ps_index)):
        index = np.argmax(ps_size)
        job_index = ps_index[index]
        ps_node = solution[job_index][0][0]
        worker_node = [solution[job_index][j][0][0] for j in range(1, len(solution[job_index]))]
        if fa <= fb:
            a_init.append(job_index)
            fa += sum_traffic[job_index]
            for j in range(0, len(solution[job_index]) - 1):
                tba[ps_node][worker_node[j]] = 1
                tba[worker_node[j]][ps_node] = 1
        else:
            b_init.append(job_index)
            fb += sum_traffic[job_index]
            for j in range(0, len(solution[job_index]) - 1):
                tbb[ps_node][worker_node[j]] = 1
                tbb[worker_node[j]][ps_node] = 1
        ps_size[job_index] = 0
    for i in range(0, len(ep_index)):
        index = np.argmax(ep_num)
        job_index = ep_index[index]
        worker_node = [solution[job_index][j][0][0] for j in range(0, len(solution[job_index]))]
        score_a = 0
        score_b = 0
        for p in worker_node:
            for q in worker_node:
                if p > q and tba[p][q] == 1:
                    score_a += 1
                if p > q and tbb[p][q] == 1:
                    score_b += 1
        if score_a <= score_b:
            a_init.append(job_index)
            fa += sum_traffic[job_index]
            for p in worker_node:
                for q in worker_node:
                    if p > q:
                        tba[p][q] = 1
                        tba[q][p] = 1
        else:
            b_init.append(job_index)
            fb += sum_traffic[job_index]
            for p in worker_node:
                for q in worker_node:
                    if p > q:
                        tbb[p][q] = 1
                        tbb[q][p] = 1
    for i in range(0, len(rar_index)):
        index = np.argmax(rar_size)
        job_index = rar_index[index]
        if fa <= fb:
            a_init.append(job_index)
            fa += sum_traffic[job_index]
        else:
            b_init.append(job_index)
            fb += sum_traffic[job_index]
    return a_init, b_init


def cut_job(sum_traffic, threshold):
    sum_traffic = np.array([sum_traffic[i] for i in range(0, len(sum_traffic))])
    sum_all = sum(sum_traffic)
    sum_main = sum_all * threshold
    sum_traffic_copy = copy.deepcopy(sum_traffic)
    count_traffic = 0
    main_job = []
    for i in range(0, len(sum_traffic_copy)):
        if count_traffic <= sum_main:
            index = np.argmax(sum_traffic_copy)
            main_job.append(index)
            count_traffic += sum_traffic_copy[index]
            sum_traffic_copy[index] = 0
        else:
            return main_job


def init_group_new(job_set, sum_traffic, solution, threshold):
    c = np.zeros([len(job_set), len(job_set)])
    adjoin_a = []
    adjoin_b = []
    for i in range(0, len(job_set)):
        for j in range(i, len(job_set)):
            if job_set[i][3] != 1:
                worker_i = [solution[i][k1][0][0] for k1 in range(0, len(solution[i]))]
            else:
                worker_i = [solution[i][k1][0][0] for k1 in range(1, len(solution[i]))]
            if job_set[j][3] != 1:
                worker_j = [solution[j][k1][0][0] for k1 in range(0, len(solution[j]))]
            else:
                worker_j = [solution[j][k1][0][0] for k1 in range(1, len(solution[j]))]
            score = 0
            for n in worker_i:
                if n in worker_j:
                    score += 1
            c[i][j] = score
            c[j][i] = score
    main_job = cut_job(sum_traffic, threshold)
    min_job = [job_set[i][0] for i in range(0, len(job_set)) if int(job_set[i][0]) not in main_job]
    fa = 0
    fb = 0
    a_init = []
    b_init = []
    for i in main_job:
        if fa <= fb:
            a_init.append(i)
            fa += sum_traffic[i]
        else:
            b_init.append(i)
            fb += sum_traffic[i]
    adjoin_a = copy.deepcopy(min_job)
    adjoin_b = copy.deepcopy(min_job)
    """
    for i in min_job:
        score_a = sum([c[i][j] for j in a_init])
        score_b = sum([c[i][j] for j in b_init])
        if score_a <= score_b:
            a_init.append(i)
        else:
            b_init.append(i)
    """
    score_a = 0
    score_b = 0
    for i in a_init:
        for j in a_init:
            if i > j:
                score_a += c[i][j]
    for i in b_init:
        for j in b_init:
            if i > j:
                score_b += c[i][j]
    while len(min_job) > 0:
        if score_a < score_b:
            adjoin_a = copy.deepcopy(min_job)
            adjoin_weight = []
            for i in adjoin_a:
                adjoin_weight.append(sum([c[i][j] for j in a_init]))
            min_index = np.argmax(adjoin_weight)
            job_index = adjoin_a[min_index]
            a_init.append(job_index)
            score_a += adjoin_weight[min_index]
            fa += sum_traffic[job_index]
        else:
            adjoin_weight = []
            adjoin_b = copy.deepcopy(min_job)
            for i in adjoin_b:
                adjoin_weight.append(sum([c[i][j] for j in b_init]))
            min_index = np.argmax(adjoin_weight)
            job_index = adjoin_b[min_index]
            b_init.append(job_index)
            score_b += adjoin_weight[min_index]
            fb += sum_traffic[job_index]
        min_job = [i for i in min_job if i != job_index]
    print(score_a, score_b, np.sum(c))
    return a_init, b_init, c


def tpe_js(solution, undeploy, job_set, pod_num, unit_gpu, per_oxc_port, b_tor, b_oxc, t_recon, gpu_usage, gpu_flops):
    train_time = [train_server(job_set[i][2], job_set[i][1], gpu_flops, gpu_usage, job_set[i][4] * job_set[i][5]) for i
                  in range(0, len(job_set))]
    if len(undeploy) > 1:
        return -1
    a = [job_set[i][0] for i in range(0, len(job_set))]
    single_link, sum_traffic = traffic_count(job_set)
    t = round_time(a, [], job_set, b_tor, b_oxc, solution, pod_num, single_link, per_oxc_port, train_time, t_recon)
    return t


def round_time(a, b, job_set, b_tor, b_oxc, solution, pod_num, single_link, per_oxc_port, train_time, t_recon):
    traffic_push_a, tor_push_a, traffic_pull_a, tor_pull_a = traffic_topo(a, job_set, solution, pod_num, single_link)
    traffic_push_b, tor_push_b, traffic_pull_b, tor_pull_b = traffic_topo(b, job_set, solution, pod_num, single_link)
    t_iter_push_a, topo_iter_push_a = oxc_count(traffic_push_a, tor_push_a, per_oxc_port, b_tor, b_oxc)
    t_iter_pull_a, topo_iter_pull_a = oxc_count(traffic_pull_a, tor_pull_a, per_oxc_port, b_tor, b_oxc)
    t_iter_push_b, topo_iter_push_b = oxc_count(traffic_push_b, tor_push_b, per_oxc_port, b_tor, b_oxc)
    t_iter_pull_b, topo_iter_pull_b = oxc_count(traffic_pull_b, tor_pull_b, per_oxc_port, b_tor, b_oxc)
    train_a = max([0] + [train_time[j] for j in a])
    train_a_agg = max([0] + [train_time[j] for j in a if job_set[j][3] != 1])
    train_b = max([0] + [train_time[j] for j in b])
    train_b_agg = max([0] + [train_time[j] for j in b if job_set[j][3] != 1])
    t_recon_1 = adjoin_topo(topo_iter_push_a, topo_iter_push_b, per_oxc_port) * t_recon
    t_recon_2 = adjoin_topo(topo_iter_push_b, topo_iter_pull_a, per_oxc_port) * t_recon
    t_recon_3 = adjoin_topo(topo_iter_pull_a, topo_iter_pull_b, per_oxc_port) * t_recon
    t_recon_4 = adjoin_topo(topo_iter_pull_b, topo_iter_push_a, per_oxc_port) * t_recon
    t_1 = max(train_a, t_iter_pull_b + t_recon_4)
    t_2 = max(train_b, t_iter_push_a + t_recon_1)
    t_3 = max(train_a_agg, t_iter_push_b + t_recon_2)
    t_4 = max(train_b_agg, t_iter_pull_a + t_recon_3)
    return t_1 + t_2 + t_3 + t_4


def mountain_climb(solution, undeploy, job_set, pod_num, unit_gpu, per_oxc_port, b_tor, b_oxc, t_recon, gpu_usage,
                   gpu_flops, threshold, iter_lim, change_rate):
    """
    爬山法迭代最优解
    :param undeploy:
    :param solution:
    :param change_rate:
    :param iter_lim:
    :param threshold:
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
    list_t = []
    single_link, sum_traffic = traffic_count(job_set)
    train_time = [train_server(job_set[i][2], job_set[i][1], gpu_flops, gpu_usage, job_set[i][4] * job_set[i][5]) for i
                  in range(0, len(job_set))]
    print("finish deploy")
    if len(undeploy) > 0:
        return -1, -1, -1, -1
    a, b, c = init_group_new(job_set, sum_traffic, solution, threshold)
    t_iter = round_time(a, b, job_set, b_tor, b_oxc, solution, pod_num, single_link, per_oxc_port, train_time, t_recon)
    print(t_iter)
    end_4 = time.time()
    t_testbench = t_iter + 0
    list_t.append(t_testbench)
    k = 0
    flag = 0
    print("init")
    c_queue = np.zeros(len(job_set))
    for i in range(0, len(c_queue)):
        if i in a:
            c_queue[i] = sum([c[i][j] for j in b])
        else:
            c_queue[i] = sum([c[i][j] for j in a])
    change_num = int(len(job_set) * change_rate)
    while 1:
        # print("iteration" + str(k))
        if k == iter_lim:
            return t_iter, t_testbench, (t_testbench - t_iter) / t_testbench, end_4
        t_iter_list = np.zeros(len(a) + len(b))
        k += 1
        print(k)
        c_queue_copy = copy.deepcopy(c_queue)
        c_change = []
        for i in range(0, change_num):
            index = np.argmin(c_queue_copy)
            c_change.append(index)
            c_queue_copy[index] = np.Inf
        """
        if flag == len(a + b):
            return t_iter
        """
        for i in c_change:
            ai = copy.deepcopy(a)
            bi = copy.deepcopy(b)
            if i in a:
                ai = [j for j in a if j != i]
                bi += [i]
            else:
                bi = [j for j in b if j != i]
                ai += [i]
            t_iter_list[i] = round_time(ai, bi, job_set, b_tor, b_oxc, solution, pod_num, single_link, per_oxc_port,
                                        train_time, t_recon)
            # print(i, t_iter_list[i])
        # print(t_iter)
        """
        if min(t_iter_list) == np.inf:
            b += [a[0]]
            a = [j for j in a if j != a[0]]
        """
        if np.min(t_iter_list) > t_iter:
            return t_iter, t_testbench, (t_testbench - t_iter) / t_testbench, end_4
        else:
            index = np.argmin(t_iter_list)
            if index in a:
                a = [j for j in a if j != index]
                b += [index]
            else:
                b = [j for j in b if j != index]
                a += [index]
            t_iter_list = np.array([i for i in t_iter_list if i != 0])
            t_iter = min(t_iter_list)
            flag_l = 0
            if len(list_t) > 3:
                list_t_copy = [list_t[j] for j in range(len(list_t) - 3, len(list_t))]
                for j in range(1, len(list_t_copy)):
                    list_t_copy[j] -= list_t_copy[0]
                for j in range(1, len(list_t_copy)):
                    if list_t_copy[j] != 0:
                        flag_l = 1
                if flag_l == 0:
                    return t_iter, t_testbench, (t_testbench - t_iter) / t_testbench, end_4
            list_t.append(t_iter)


def round_time_pjs(a, b, job_set, b_tor, b_oxc, solution, pod_num, single_link, per_oxc_port, train_time, t_recon):
    traffic_push_a, tor_push_a, traffic_pull_a, tor_pull_a = traffic_topo(a, job_set, solution, pod_num, single_link)
    traffic_push_b, tor_push_b, traffic_pull_b, tor_pull_b = traffic_topo(b, job_set, solution, pod_num, single_link)
    t_push_a = time_count(b_oxc, traffic_push_a, pod_num, tor_push_a, b_tor)
    t_pull_a = time_count(b_oxc, traffic_pull_a, pod_num, tor_pull_a, b_tor)
    t_push_b = time_count(b_oxc, traffic_push_b, pod_num, tor_push_b, b_tor)
    t_pull_b = time_count(b_oxc, traffic_pull_b, pod_num, tor_pull_b, b_tor)
    train_a = max([train_time[i] for i in a] + [0])
    train_b = max([train_time[i] for i in b] + [0])
    train_a_agg = max([train_time[i] for i in a if job_set[i][3] != 1] + [0])
    train_b_agg = max([train_time[i] for i in b if job_set[i][3] != 1] + [0])
    t_1 = max(train_a, t_pull_b)
    t_2 = max(train_b, t_push_a)
    t_3 = max(train_a_agg, t_push_b)
    t_4 = max(train_b_agg, t_pull_a)
    return t_1 + t_2 + t_3 + t_4


def mountain_pjs(solution, undeploy, job_set, pod_num, unit_gpu, per_oxc_port, b_tor, b_oxc, t_recon, gpu_usage,
                 gpu_flops, threshold, iter_lim, change_rate):
    list_t = []
    train_time = [train_server(job_set[i][2], job_set[i][1], gpu_flops, gpu_usage, job_set[i][4] * job_set[i][5]) for i
                  in range(0, len(job_set))]
    single_link, sum_traffic = traffic_count(job_set)
    print("finish deploy")
    if len(undeploy) > 0:
        return -1, -1, -1, -1
    a, b, c = init_group_new(job_set, sum_traffic, solution, threshold)
    t_iter = round_time_pjs(a, b, job_set, b_tor, b_oxc, solution, pod_num, single_link, per_oxc_port, train_time,
                            t_recon)
    print(t_iter)
    end_5 = time.time()
    t_testbench = t_iter + 0
    list_t.append(t_testbench)
    k = 0
    flag = 0
    print("init")
    c_queue = np.zeros(len(job_set))
    for i in range(0, len(c_queue)):
        if i in a:
            c_queue[i] = sum([c[i][j] for j in b])
        else:
            c_queue[i] = sum([c[i][j] for j in a])
    change_num = int(len(job_set) * change_rate)
    while 1:
        # print("iteration" + str(k))
        if k == iter_lim:
            return t_iter, t_testbench, (t_testbench - t_iter) / t_testbench, end_5
        t_iter_list = np.zeros(len(a) + len(b))
        k += 1
        print(k)
        c_queue_copy = copy.deepcopy(c_queue)
        c_change = []
        for i in range(0, change_num):
            index = np.argmin(c_queue_copy)
            c_change.append(index)
            c_queue_copy[index] = np.Inf
        """
        if flag == len(a + b):
            return t_iter
        """
        for i in c_change:
            ai = copy.deepcopy(a)
            bi = copy.deepcopy(b)
            if i in a:
                ai = [j for j in a if j != i]
                bi += [i]
            else:
                bi = [j for j in b if j != i]
                ai += [i]
            t_iter_list[i] = round_time_pjs(ai, bi, job_set, b_tor, b_oxc, solution, pod_num, single_link, per_oxc_port,
                                            train_time, t_recon)
            # print(i, t_iter_list[i])
        # print(t_iter)
        t_iter_list = np.array([i for i in t_iter_list if i != 0])
        if np.min(t_iter_list) > t_iter:
            return t_iter, t_testbench, (t_testbench - t_iter) / t_testbench, end_5
        else:
            index = np.argmin(t_iter_list)
            if index in a:
                a = [j for j in a if j != index]
                b += [index]
            else:
                b = [j for j in b if j != index]
                a += [index]
            t_iter = min(t_iter_list)
            flag_l = 0
            if len(list_t) > 3:
                list_t_copy = [list_t[j] for j in range(len(list_t) - 3, len(list_t))]
                for j in range(1, len(list_t_copy)):
                    list_t_copy[j] -= list_t_copy[0]
                for j in range(1, len(list_t_copy)):
                    if list_t_copy[j] != 0:
                        flag_l = 1
                if flag_l == 0:
                    return t_iter, t_testbench, (t_testbench - t_iter) / t_testbench, end_5
            list_t.append(t_iter)


def time_count(b_oxc, traffic, pod_num, tor_traffic, b_tor):
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
    all_job = np.array([job_set[i][0] for i in range(0, len(job_set))])
    traffic_matrix_push = np.zeros([pod_num, pod_num])
    tor_traffic_push = np.zeros(pod_num)
    traffic_matrix_pull = np.zeros([pod_num, pod_num])
    tor_traffic_pull = np.zeros(pod_num)
    traffic_matrix_push_all = np.zeros([pod_num, pod_num])
    traffic_matrix_pull_all = np.zeros([pod_num, pod_num])
    for i in all_job:
        if job_set[i][3] == 0:
            continue
        elif job_set[i][3] == 1:
            ps_node = local_solution[i][0][0]
            worker_node = [local_solution[i][j][0][0] for j in range(1, len(local_solution[i]))]
            for j in worker_node:
                traffic_matrix_push_all[j][ps_node] += traffic_single_link[i]
                traffic_matrix_pull_all[ps_node][j] += traffic_single_link[i]
        elif job_set[i][3] == 2:
            worker_node = [local_solution[i][j][0][0] for j in range(0, len(local_solution[i]))]
            for p in worker_node:
                for q in worker_node:
                    if p != q:
                        traffic_matrix_push_all[p][q] += traffic_single_link[i]
                        traffic_matrix_pull_all[p][q] += traffic_single_link[i]
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
            link_bool_a_1 = (traffic_matrix_push_all > 0).astype(int)
        else:
            link_bool_a_1 = (traffic_matrix_pull_all > 0).astype(int)
        for i in ring_a:
            ring_oxc_traffic, ring_tor_traffic = (
                connect_link_ring(link_bool_a_1, traffic_single_link[i], local_solution[i], job_set[i]))
            link_bool_a_1 = (ring_oxc_traffic > 0).astype(int)
            if n == 0:
                traffic_matrix_push += ring_oxc_traffic
                traffic_matrix_push_all += ring_tor_traffic
                tor_traffic_push += ring_tor_traffic
            else:
                traffic_matrix_pull += ring_oxc_traffic
                traffic_matrix_pull_all += ring_oxc_traffic
                tor_traffic_pull += ring_tor_traffic
    return traffic_matrix_push, tor_traffic_push, traffic_matrix_pull, tor_traffic_pull


def deploy_server(group, job_set, group_traffic_single_link, group_traffic_size, pod, pod_gpu, oxc_tor_bandwidth_rate,
                  seg_pod_num):
    local_solution = np.empty(len(group), dtype=object)
    reverse_gpu = np.array([pod_gpu for _ in range(0, pod)])
    seg_num = int(pod / seg_pod_num)
    seg_pod_reserve = np.array([seg_pod_num * pod_gpu for _ in range(0, seg_num)])
    unit_gpu_num = np.zeros(len(group))
    for i1 in range(0, len(job_set)):
        unit_gpu_num[i1] = job_set[i1][5]
    undeploy = []
    for i1 in range(0, len(job_set)):
        if job_set[i1][3] != 1:
            available_seg = [n for n in range(0, seg_num) if seg_pod_reserve[n] >= unit_gpu_num[i1] * job_set[i1][4]]
            if len(available_seg) == 0:
                undeploy.append(i1)
            seg_index = random.choice(available_seg)
            worker_node = []
            for j in range(0, job_set[i1][4]):
                worker_node_unit = []
                seg_reserve = reverse_gpu[int(seg_index * seg_pod_num): int(seg_index * seg_pod_num + seg_pod_num)]
                max_index = np.argmax(seg_reserve) + seg_index * seg_pod_num
                if reverse_gpu[max_index] < unit_gpu_num[i1]:
                    for i in range(0, len(worker_node)):
                        reverse_gpu[worker_node[i][0][0]] += unit_gpu_num[i1]
                        seg_pod_reserve[seg_index] += unit_gpu_num[i1]
                        undeploy.append(i1)
                    worker_node = []
                    break
                worker_node_unit.append((max_index, unit_gpu_num[i1]))
                seg_reserve[max_index - seg_index * seg_pod_num] -= unit_gpu_num[i1]
                worker_node.append(worker_node_unit)
            local_solution[i1] = worker_node
        else:
            available_seg = [n for n in range(0, seg_num) if
                             seg_pod_reserve[n] >= unit_gpu_num[i1] * job_set[i1][4] + 1]
            if len(available_seg) == 0:
                undeploy.append(i1)
            seg_index = random.choice(available_seg)
            worker_node = []
            for n in range(0, seg_pod_num):
                if reverse_gpu[int(seg_index * seg_pod_num + n)] >= 1:
                    worker_node.append((seg_index * seg_pod_num + n, 1))
                    reverse_gpu[int(seg_index * seg_pod_num + n)] -= 1
                    break
            for j in range(0, job_set[i1][4]):
                worker_node_unit = []
                seg_reserve = reverse_gpu[int(seg_index * seg_pod_num): int(seg_index * seg_pod_num + seg_pod_num)]
                max_index = np.argmax(seg_reserve) + seg_index * seg_pod_num
                if reverse_gpu[max_index] < unit_gpu_num[i1]:
                    for i in range(0, len(worker_node)):
                        if i == 0:
                            reverse_gpu[worker_node[i][0]] += 1
                            seg_pod_reserve[seg_index] += 1
                            continue
                        reverse_gpu[worker_node[i][0][0]] += unit_gpu_num[i1]
                        seg_pod_reserve[seg_index] += unit_gpu_num[i1]
                        undeploy.append(i1)
                    worker_node = []
                    break
                worker_node_unit.append((max_index, unit_gpu_num[i1]))
                seg_reserve[max_index - seg_index * seg_pod_num] -= unit_gpu_num[i1]
                worker_node.append(worker_node_unit)
            local_solution[i1] = worker_node
    return local_solution, undeploy


"""
def deploy_server(group, job_set, group_traffic_single_link, group_traffic_size, pod, pod_gpu, oxc_tor_bandwidth_rate,
                  seg_pod_num):
    各组并行单元的部署方案生成，产生拓扑，注意连接选择半双工，不使用INC。部署方案改为：将所有 pod 分小组，每个业务只放进一个小组
    :return:
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
        
            for i in range(0, pod):
                if i == ps_node:
                    ava_pod.append(i)
                elif link_bool_matrix[i][ps_node] == 1:
                    ava_pod.append(i)
                elif degree[i] < oxc_num and degree[ps_node] < oxc_num:
                    ava_pod.append(i)
            
            # output_traffic[ps_node] += job_set[job_index][4] * traffic
            output_tor[ps_node] += job_set[job_index][4] * traffic
            # output_oxc[ps_node] += job_set[job_index][4] * traffic
            reverse_gpu[ps_node] -= 1
            for i in range(0, job_set[job_index][4]):
                unit_local = []
                # print('unit' + str(i + 1))
                # ava_pod = [j for j in ava_pod if unit_gpu_num[job_index] <= reverse_gpu[j]]
                output_traffic_copy = np.array([max(output_tor[j], output_oxc[j] / oxc_tor_bandwidth_rate) if
                                                reverse_gpu[j] > job_set[job_index][5] else np.inf for j in
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
            
            if link_bool_matrix[ps_node][min_index] == 0:
                degree[ps_node] += 1
                degree[min_index] += 1
                if degree[ps_node] == oxc_num:
                    ava_pod = [j for j in ava_pod if link_bool_matrix[ps_node][j] == 1 or
                               link_job[ps_node][j] == 1]
            
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
"""


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
        para = [2.46, 6.42, 17.22, 21.2]
        job_para = para[job_size]
        # context = 8
        # job_para = np.random.choice(np.array([1, 3, 8, 70]), p=np.array([0.25, 0.25, 0.25]))
        job_type = np.random.choice(np.array([0, 1, 2]), p=np.array([1 / 3, 1 / 3, 1 / 3]))
        parallel = np.random.choice(np.array([2, 4, 6]), p=np.array([1 / 3, 1 / 3, 1 / 3]))
        # token_per_batch =
        # np.random.choice(np.array([128, 256, 512, 1024, 2048]), p=np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        if job_size < 2:
            unit_gpu = np.random.randint(2, 5)
        else:
            unit_gpu = np.random.randint(5, 10)
        job_default = [i, job_para, 1, job_type, parallel, unit_gpu]
        job_context_2 = [i, job_para, 2, job_type, parallel, unit_gpu]
        job_context_3 = [i, job_para, 4, job_type, parallel, unit_gpu]
        job_context_4 = [i, job_para, 8, job_type, parallel, unit_gpu]
        job_context_5 = [i, job_para, 16, job_type, parallel, unit_gpu]

        job_set1.append(job_default)
        job_set2.append(job_context_2)
        job_set3.append(job_context_3)
        job_set4.append(job_context_4)
        job_set5.append(job_context_5)
    return job_set1, job_set2, job_set3, job_set4, job_set5


for k in range(0, 10):
    usage = 0.4
    iter_num = 10
    flop = 275
    info = generate_job(100)
    job = info[2]
    job1 = job[0:100]
    all_job_index = [job1[i][0] for i in range(0, len(job1))]
    single_link_out, sum_traffic_out = traffic_count(job1)
    solution_out, undeploy_out = deploy_server(all_job_index, job1, single_link_out, sum_traffic_out, 64, 64, 1, 8)
    start1 = time.time()
    t1, t2, _, end4 = mountain_climb(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64,
                                     per_oxc_port=16, b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop,
                                     threshold=0.9, iter_lim=iter_num, change_rate=0.3)
    end1 = time.time()
    start2 = time.time()
    t3 = tpe_js(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64, per_oxc_port=16, b_tor=1600,
                b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop)
    end2 = time.time()
    start3 = time.time()
    t4, t5, _, end5 = mountain_pjs(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64, per_oxc_port=16,
                                   b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop, threshold=0.9,
                                   iter_lim=iter_num, change_rate=0.3)
    end3 = time.time()
    data = [t2, t1, t3, t5, t4, end4 - start1, end1 - start1, end2 - start2, end5 - start3, end3 - start3]
    sheet.append(data)

    t1, t2, _, end4 = mountain_climb(solution_out[0:80], undeploy_out, job1[0:80], pod_num=64, unit_gpu=64,
                                     per_oxc_port=16, b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop,
                                     threshold=0.9, iter_lim=iter_num, change_rate=0.3)
    end1 = time.time()
    start2 = time.time()
    t3 = tpe_js(solution_out[0:80], undeploy_out, job1[0:80], pod_num=64, unit_gpu=64, per_oxc_port=16, b_tor=1600, b_oxc=100,
                t_recon=0.1, gpu_usage=usage, gpu_flops=flop)
    end2 = time.time()
    start3 = time.time()
    t4, t5, _, end5 = mountain_pjs(solution_out[0:80], undeploy_out, job1[0:80], pod_num=64, unit_gpu=64, per_oxc_port=16,
                                   b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop, threshold=0.9,
                                   iter_lim=iter_num, change_rate=0.3)
    end3 = time.time()
    data = [t2, t1, t3, t5, t4, end4 - start1, end1 - start1, end2 - start2, end5 - start3, end3 - start3]
    sheet.append(data)

    t1, t2, _, end4 = mountain_climb(solution_out[0:60], undeploy_out, job1[0:60], pod_num=64, unit_gpu=64,
                                     per_oxc_port=16, b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop,
                                     threshold=0.9, iter_lim=iter_num, change_rate=0.3)
    end1 = time.time()
    start2 = time.time()
    t3 = tpe_js(solution_out[0:60], undeploy_out, job1[0:60], pod_num=64, unit_gpu=64, per_oxc_port=16, b_tor=1600, b_oxc=100,
                t_recon=0.1, gpu_usage=usage, gpu_flops=flop)
    end2 = time.time()
    start3 = time.time()
    t4, t5, _, end5 = mountain_pjs(solution_out[0:60], undeploy_out, job1[0:60], pod_num=64, unit_gpu=64, per_oxc_port=16,
                                   b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop, threshold=0.9,
                                   iter_lim=iter_num, change_rate=0.3)
    end3 = time.time()
    data = [t2, t1, t3, t5, t4, end4 - start1, end1 - start1, end2 - start2, end5 - start3, end3 - start3]
    sheet.append(data)

    t1, t2, _, end4 = mountain_climb(solution_out[0:40], undeploy_out, job1[0:40], pod_num=64, unit_gpu=64,
                                     per_oxc_port=16, b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop,
                                     threshold=0.9, iter_lim=iter_num, change_rate=0.3)
    end1 = time.time()
    start2 = time.time()
    t3 = tpe_js(solution_out[0:40], undeploy_out, job1[0:40], pod_num=64, unit_gpu=64, per_oxc_port=16, b_tor=1600, b_oxc=100,
                t_recon=0.1, gpu_usage=usage, gpu_flops=flop)
    end2 = time.time()
    start3 = time.time()
    t4, t5, _, end5 = mountain_pjs(solution_out[0:40], undeploy_out, job1[0:40], pod_num=64, unit_gpu=64, per_oxc_port=16,
                                   b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop, threshold=0.9,
                                   iter_lim=iter_num, change_rate=0.3)
    end3 = time.time()
    data = [t2, t1, t3, t5, t4, end4 - start1, end1 - start1, end2 - start2, end5 - start3, end3 - start3]
    sheet.append(data)

    t1, t2, _, end4 = mountain_climb(solution_out[0:20], undeploy_out, job1[0:20], pod_num=64, unit_gpu=64,
                                     per_oxc_port=16, b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop,
                                     threshold=0.9, iter_lim=iter_num, change_rate=0.3)
    end1 = time.time()
    start2 = time.time()
    t3 = tpe_js(solution_out[0:20], undeploy_out, job1[0:20], pod_num=64, unit_gpu=64, per_oxc_port=16, b_tor=1600, b_oxc=100,
                t_recon=0.1, gpu_usage=usage, gpu_flops=flop)
    end2 = time.time()
    start3 = time.time()
    t4, t5, _, end5 = mountain_pjs(solution_out[0:20], undeploy_out, job1[0:20], pod_num=64, unit_gpu=64, per_oxc_port=16,
                                   b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop, threshold=0.9,
                                   iter_lim=iter_num, change_rate=0.3)
    end3 = time.time()
    data = [t2, t1, t3, t5, t4, end4 - start1, end1 - start1, end2 - start2, end5 - start3, end3 - start3]
    sheet.append(data)
    sheet.append([])

    start1 = time.time()
    t1, t2, _, end4 = mountain_climb(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64,
                                     per_oxc_port=16, b_tor=800, b_oxc=50, t_recon=0.1, gpu_usage=usage, gpu_flops=flop,
                                     threshold=0.9, iter_lim=iter_num, change_rate=0.3)
    end1 = time.time()
    start2 = time.time()
    t3 = tpe_js(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64, per_oxc_port=16, b_tor=800,
                b_oxc=50, t_recon=0.1, gpu_usage=usage, gpu_flops=flop)
    end2 = time.time()
    start3 = time.time()
    t4, t5, _, end5 = mountain_pjs(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64, per_oxc_port=16,
                                   b_tor=800, b_oxc=50, t_recon=0.1, gpu_usage=usage, gpu_flops=flop, threshold=0.9,
                                   iter_lim=iter_num, change_rate=0.3)
    end3 = time.time()
    data = [t2, t1, t3, t5, t4, end4 - start1, end1 - start1, end2 - start2, end5 - start3, end3 - start3]
    sheet.append(data)

    start1 = time.time()
    t1, t2, _, end4 = mountain_climb(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64,
                                     per_oxc_port=16, b_tor=1200, b_oxc=75, t_recon=0.1, gpu_usage=usage, gpu_flops=flop,
                                     threshold=0.9, iter_lim=iter_num, change_rate=0.3)
    end1 = time.time()
    start2 = time.time()
    t3 = tpe_js(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64, per_oxc_port=16, b_tor=1200,
                b_oxc=75, t_recon=0.1, gpu_usage=usage, gpu_flops=flop)
    end2 = time.time()
    start3 = time.time()
    t4, t5, _, end5 = mountain_pjs(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64, per_oxc_port=16,
                                   b_tor=1200, b_oxc=75, t_recon=0.1, gpu_usage=usage, gpu_flops=flop, threshold=0.9,
                                   iter_lim=iter_num, change_rate=0.3)
    end3 = time.time()
    data = [t2, t1, t3, t5, t4, end4 - start1, end1 - start1, end2 - start2, end5 - start3, end3 - start3]
    sheet.append(data)

    start1 = time.time()
    t1, t2, _, end4 = mountain_climb(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64,
                                     per_oxc_port=16, b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop,
                                     threshold=0.9, iter_lim=iter_num, change_rate=0.3)
    end1 = time.time()
    start2 = time.time()
    t3 = tpe_js(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64, per_oxc_port=16, b_tor=1600,
                b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop)
    end2 = time.time()
    start3 = time.time()
    t4, t5, _, end5 = mountain_pjs(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64, per_oxc_port=16,
                                   b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop, threshold=0.9,
                                   iter_lim=iter_num, change_rate=0.3)
    end3 = time.time()
    data = [t2, t1, t3, t5, t4, end4 - start1, end1 - start1, end2 - start2, end5 - start3, end3 - start3]
    sheet.append(data)

    start1 = time.time()
    t1, t2, _, end4 = mountain_climb(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64,
                                     per_oxc_port=16, b_tor=2000, b_oxc=125, t_recon=0.1, gpu_usage=usage, gpu_flops=flop,
                                     threshold=0.9, iter_lim=iter_num, change_rate=0.3)
    end1 = time.time()
    start2 = time.time()
    t3 = tpe_js(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64, per_oxc_port=16, b_tor=2000,
                b_oxc=125, t_recon=0.1, gpu_usage=usage, gpu_flops=flop)
    end2 = time.time()
    start3 = time.time()
    t4, t5, _, end5 = mountain_pjs(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64, per_oxc_port=16,
                                   b_tor=2000, b_oxc=125, t_recon=0.1, gpu_usage=usage, gpu_flops=flop, threshold=0.9,
                                   iter_lim=iter_num, change_rate=0.3)
    end3 = time.time()
    data = [t2, t1, t3, t5, t4, end4 - start1, end1 - start1, end2 - start2, end5 - start3, end3 - start3]
    sheet.append(data)

    start1 = time.time()
    t1, t2, _, end4 = mountain_climb(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64,
                                     per_oxc_port=16, b_tor=2400, b_oxc=150, t_recon=0.1, gpu_usage=usage, gpu_flops=flop,
                                     threshold=0.9, iter_lim=iter_num, change_rate=0.3)
    end1 = time.time()
    start2 = time.time()
    t3 = tpe_js(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64, per_oxc_port=16, b_tor=2400,
                b_oxc=150, t_recon=0.1, gpu_usage=usage, gpu_flops=flop)
    end2 = time.time()
    start3 = time.time()
    t4, t5, _, end5 = mountain_pjs(solution_out[0:100], undeploy_out, job1[0:100], pod_num=64, unit_gpu=64, per_oxc_port=16,
                                   b_tor=2400, b_oxc=150, t_recon=0.1, gpu_usage=usage, gpu_flops=flop, threshold=0.9,
                                   iter_lim=iter_num, change_rate=0.3)
    end3 = time.time()
    data = [t2, t1, t3, t5, t4, end4 - start1, end1 - start1, end2 - start2, end5 - start3, end3 - start3]
    sheet.append(data)
    sheet.append([])

    start1 = time.time()
    t1, t2, _, end4 = mountain_climb(solution_out[0:100], undeploy_out, info[0], pod_num=64, unit_gpu=64,
                                     per_oxc_port=16, b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop,
                                     threshold=0.9, iter_lim=iter_num, change_rate=0.3)
    end1 = time.time()
    start2 = time.time()
    t3 = tpe_js(solution_out[0:100], undeploy_out, info[0], pod_num=64, unit_gpu=64, per_oxc_port=16, b_tor=1600,
                b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop)
    end2 = time.time()
    start3 = time.time()
    t4, t5, _, end5 = mountain_pjs(solution_out[0:100], undeploy_out, info[0], pod_num=64, unit_gpu=64, per_oxc_port=16,
                                   b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop, threshold=0.9,
                                   iter_lim=iter_num, change_rate=0.3)
    end3 = time.time()
    data = [t2, t1, t3, t5, t4, end4 - start1, end1 - start1, end2 - start2, end5 - start3, end3 - start3]
    sheet.append(data)


    start1 = time.time()
    t1, t2, _, end4 = mountain_climb(solution_out[0:100], undeploy_out, info[1], pod_num=64, unit_gpu=64,
                                     per_oxc_port=16, b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop,
                                     threshold=0.9, iter_lim=iter_num, change_rate=0.3)
    end1 = time.time()
    start2 = time.time()
    t3 = tpe_js(solution_out[0:100], undeploy_out, info[1], pod_num=64, unit_gpu=64, per_oxc_port=16, b_tor=1600,
                b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop)
    end2 = time.time()
    start3 = time.time()
    t4, t5, _, end5 = mountain_pjs(solution_out[0:100], undeploy_out, info[1], pod_num=64, unit_gpu=64, per_oxc_port=16,
                                   b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop, threshold=0.9,
                                   iter_lim=iter_num, change_rate=0.3)
    end3 = time.time()
    data = [t2, t1, t3, t5, t4, end4 - start1, end1 - start1, end2 - start2, end5 - start3, end3 - start3]
    sheet.append(data)

    start1 = time.time()
    t1, t2, _, end4 = mountain_climb(solution_out[0:100], undeploy_out, info[2], pod_num=64, unit_gpu=64,
                                     per_oxc_port=16, b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop,
                                     threshold=0.9, iter_lim=iter_num, change_rate=0.3)
    end1 = time.time()
    start2 = time.time()
    t3 = tpe_js(solution_out[0:100], undeploy_out, info[2], pod_num=64, unit_gpu=64, per_oxc_port=16, b_tor=1600,
                b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop)
    end2 = time.time()
    start3 = time.time()
    t4, t5, _, end5 = mountain_pjs(solution_out[0:100], undeploy_out, info[2], pod_num=64, unit_gpu=64, per_oxc_port=16,
                                   b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop, threshold=0.9,
                                   iter_lim=iter_num, change_rate=0.3)
    end3 = time.time()
    data = [t2, t1, t3, t5, t4, end4 - start1, end1 - start1, end2 - start2, end5 - start3, end3 - start3]
    sheet.append(data)

    start1 = time.time()
    t1, t2, _, end4 = mountain_climb(solution_out[0:100], undeploy_out, info[3], pod_num=64, unit_gpu=64,
                                     per_oxc_port=16, b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop,
                                     threshold=0.9, iter_lim=iter_num, change_rate=0.3)
    end1 = time.time()
    start2 = time.time()
    t3 = tpe_js(solution_out[0:100], undeploy_out, info[3], pod_num=64, unit_gpu=64, per_oxc_port=16, b_tor=1600,
                b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop)
    end2 = time.time()
    start3 = time.time()
    t4, t5, _, end5 = mountain_pjs(solution_out[0:100], undeploy_out, info[3], pod_num=64, unit_gpu=64, per_oxc_port=16,
                                   b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop, threshold=0.9,
                                   iter_lim=iter_num, change_rate=0.3)
    end3 = time.time()
    data = [t2, t1, t3, t5, t4, end4 - start1, end1 - start1, end2 - start2, end5 - start3, end3 - start3]
    sheet.append(data)

    start1 = time.time()
    t1, t2, _, end4 = mountain_climb(solution_out[0:100], undeploy_out, info[4], pod_num=64, unit_gpu=64,
                                     per_oxc_port=16, b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop,
                                     threshold=0.9, iter_lim=iter_num, change_rate=0.3)
    end1 = time.time()
    start2 = time.time()
    t3 = tpe_js(solution_out[0:100], undeploy_out, info[4], pod_num=64, unit_gpu=64, per_oxc_port=16, b_tor=1600,
                b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop)
    end2 = time.time()
    start3 = time.time()
    t4, t5, _, end5 = mountain_pjs(solution_out[0:100], undeploy_out, info[4], pod_num=64, unit_gpu=64, per_oxc_port=16,
                                   b_tor=1600, b_oxc=100, t_recon=0.1, gpu_usage=usage, gpu_flops=flop, threshold=0.9,
                                   iter_lim=iter_num, change_rate=0.3)
    end3 = time.time()
    data = [t2, t1, t3, t5, t4, end4 - start1, end1 - start1, end2 - start2, end5 - start3, end3 - start3]
    sheet.append(data)
    sheet.append([])
    sheet.append(["---", "---", "---", "---", "---"])
    sheet.append([])

workbook.save("output.xlsx")
