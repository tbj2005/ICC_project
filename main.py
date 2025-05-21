import copy
import time

import numpy as np
import bvn
import Schedule_part
from scipy.optimize import linear_sum_assignment
import heapq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# 确定业务成环方案
def job_ring(fj, ufj, local_solution, single_traffic, sum_traffic, pod):
    """
    确定业务成环方案
    :param sum_traffic: 各业务总流量
    :param fj: 固定拓扑业务索引
    :param ufj: 环拓扑业务索引
    :param local_solution: 业务放置方案
    :param single_traffic: 业务单连接流量
    :param pod: pod 数目
    :return: 返回所有业务的流量矩阵
    """
    job_num = len(local_solution)
    link_job_index = [[[] for _ in range(pod)] for _ in range(pod)]
    data_matrix_all = np.zeros(([pod, pod]))
    data_matrix_job = np.array([np.zeros([pod, pod]) for _ in range(job_num)])
    for i in fj:
        ps_local = local_solution[i][0]
        worker = [k for k in range(pod) if local_solution[i][1][k] > 0]
        for j in worker:
            if j == ps_local:
                continue
            else:
                data_matrix_job[i][ps_local][j] += single_traffic[i]
                data_matrix_job[i][j][ps_local] += single_traffic[i]
                link_job_index[j][ps_local].append(i)
                link_job_index[ps_local][j].append(i)
        data_matrix_all += data_matrix_job[i]
    sum_traffic_ufj = np.array([sum_traffic[i] if i in ufj else 0 for i in range(len(sum_traffic))])
    for i in range(len(ufj)):
        job_index = np.argmax(sum_traffic_ufj)
        data_matrix_stuff, _ = bvn.solve_target_matrix(data_matrix_all, pod)
        bvn_compose, bvn_sum = bvn.matrix_decompose(data_matrix_all, data_matrix_stuff, pod, 0.8, - 1)
        sum_bvn = sum(bvn_compose)
        worker = [k for k in range(pod) if local_solution[job_index][1][k] > 0]
        worker_matrix = np.zeros([len(worker), len(worker)])
        for u in range(len(worker)):
            for v in range(len(worker)):
                if u != v:
                    worker_matrix[u][v] = sum_bvn[worker[u]][worker[v]]
                if u == v:
                    worker_matrix[u][v] = - np.inf
        if np.shape(worker_matrix)[0] == 1:
            data_matrix_job[job_index] = np.zeros([pod, pod])
        else:
            row_ind, col_ind = linear_sum_assignment(worker_matrix, maximize=True)
            data_matrix_single = np.zeros([pod, pod])
            for u in range(len(row_ind)):
                data_matrix_single[worker[row_ind[u]]][worker[col_ind[u]]] = single_traffic[job_index]
                link_job_index[worker[row_ind[u]]][worker[col_ind[u]]].append(job_index)
            data_matrix_job[job_index] = data_matrix_single
        sum_traffic_ufj[job_index] = -1
        data_matrix_all += data_matrix_job[job_index]
    return data_matrix_job, link_job_index


# TPE 方案
def is_strongly_connected_scipy(adj_matrix):
    """
    使用SciPy检查强连通性（推荐）
    """
    n = adj_matrix.shape[0]
    if n == 0:
        return True

    # 转换为稀疏矩阵（仅需连接是否存在）
    graph = csr_matrix((adj_matrix > 0).astype(int))
    n_components, _ = connected_components(graph, directed=True, connection='strong')
    return n_components == 1


def sort_indices_desc(matrix, num):
    # 获取矩阵的尺寸
    n = matrix.shape[0]

    # 将矩阵展平并获取排序后的索引
    flattened = matrix.flatten()
    sorted_indices = np.argsort(flattened)[::-1]  # 降序排序索引

    # 输出前N个索引
    top_n_indices = sorted_indices[:num]

    # 将前N个一维索引转换为二维索引
    indices = np.unravel_index(top_n_indices, (n, n))

    return indices[0], indices[1]


def port_allocate(bvn_compose, port, pod):
    if len(bvn_compose) > port:
        return np.zeros([pod, pod])
    else:
        value_match = np.array([np.sum(bvn_compose[i]) / pod for i in range(len(bvn_compose))])
        match_degree = np.ones(len(bvn_compose))
        count = port - len(bvn_compose)
        while 1:
            if count == 0:
                link_matrix = np.zeros([pod, pod])
                for i in range(len(match_degree)):
                    bool_matrix = np.where(bvn_compose[i] > 0, 1, 0)
                    link_matrix += match_degree[i] * bool_matrix
                return link_matrix
            max_value, max_index = np.max(value_match), np.argmax(value_match)
            match_degree[max_index] += 1
            value_match[max_index] = np.sum(bvn_compose[max_index]) / (pod * match_degree[max_index])
            count -= 1


def max_min_weight_path_dijkstra_no_cycle(adj_matrix, start, end):
    n = adj_matrix.shape[0]
    max_min_heap = [(-float('inf'), start, [start])]  # (-当前路径最小权重, 当前节点, 路径)

    while max_min_heap:
        current_min_neg, u, path = heapq.heappop(max_min_heap)
        current_min = -current_min_neg

        if u == end:
            return current_min, path

        for v in range(n):
            if adj_matrix[u][v] > 0 and v not in path:  # 禁止重复访问节点
                new_min = min(current_min, adj_matrix[u][v])
                heapq.heappush(max_min_heap, (-new_min, v, path + [v]))

    return None, []  # 不可达


def binary_search(t_low, t_high, job_matrix, sum_matrix, link_matrix, band_per_port, traffic_size, t_threshold,
                  job_link_match):
    """
    二分查找理想时间
    :param job_link_match:
    :param t_threshold:
    :param traffic_size:
    :param sum_matrix:
    :param band_per_port:
    :param t_low:
    :param t_high:
    :param job_matrix:
    :param link_matrix:
    :return:
    """
    job_matrix_out = copy.deepcopy(job_matrix)
    sum_matrix_out = copy.deepcopy(job_matrix)
    while 1:
        if t_high - t_low <= t_threshold:
            return job_matrix_out, sum_matrix_out
        sum_matrix_edit = copy.deepcopy(sum_matrix)
        job_matrix_edit = copy.deepcopy(job_matrix)
        t_mid = (t_low + t_high) / 2
        ideal_matrix = t_mid * band_per_port * link_matrix
        delta_matrix = ideal_matrix - sum_matrix_edit
        stuff_decompose = np.where(delta_matrix > 0, delta_matrix, 0)
        reserve_decompose = np.where(delta_matrix < 0, - delta_matrix, 0)
        reverse_row, reverse_col = sort_indices_desc(reserve_decompose, np.count_nonzero(reserve_decompose))
        job_index_sort = np.argsort(- np.array(traffic_size))
        for i in range(np.count_nonzero(reserve_decompose > 0)):
            row, col = reverse_row[i], reverse_col[i]
            job_set_link = job_link_match[row][col]
            for j in range(len(job_index_sort)):
                if reserve_decompose[row][col] <= 0:
                    continue
                job_index = job_index_sort[j]
                value = job_matrix_edit[job_index][row][col]
                if job_index in job_set_link:
                    _, path = max_min_weight_path_dijkstra_no_cycle(delta_matrix, row, col)
                    flag = 0
                    for k in range(len(path) - 1):
                        if stuff_decompose[path[k]][path[k + 1]] < value:
                            flag = 1
                            break
                    if flag == 1:
                        break
                    else:
                        job_matrix_edit[job_index][row][col] = 0
                        for k in range(len(path) - 1):
                            stuff_decompose[path[k]][path[k + 1]] -= value
                            job_matrix_edit[job_index][path[k]][path[k + 1]] += value
                            sum_matrix_edit[path[k]][path[k + 1]] -= value
                        reserve_decompose[row][col] -= value
                        sum_matrix_edit[row][col] -= value
                        continue
            if reserve_decompose[row][col] > 0:
                t_low = t_mid
                break
        t_high = t_mid
        job_matrix_out = copy.deepcopy(job_matrix_edit)
        sum_matrix_out = copy.deepcopy(sum_matrix_edit)


def sub_ring_edit(matrix, pod):
    reverse_pod = [i for i in range(pod)]
    ring = []
    while 1:
        sub_ring = []
        if len(reverse_pod) == 0:
            break
        sub_ring.append(reverse_pod[0])
        while 1:
            next_node = np.argmax(matrix[sub_ring[-1]])
            if next_node in sub_ring:
                break
            else:
                sub_ring.append(next_node)
        reverse_pod = [i for i in reverse_pod if i not in sub_ring]
        ring.append(sub_ring)
    edit_matrix = np.zeros([pod, pod])
    edit_ring = []
    for i in range(len(ring)):
        edit_ring += ring[i]
    for i in range(pod - 1):
        edit_matrix[edit_ring[i]][edit_ring[i + 1]] = np.max(matrix)
    edit_matrix[edit_ring[-1]][edit_ring[0]] = np.max(matrix)
    return edit_matrix


def tpe(job_matrix, job_link_index, port, pod, num_bvn, band_per_port, single_traffic):
    """
    tpe 策略
    :param single_traffic:
    :param band_per_port:
    :param num_bvn:
    :param job_link_index: 各连接上存在业务的索引
    :param pod: pod 数目
    :param job_matrix: 总流量矩阵
    :param port: port 数目
    :return:
    """
    sum_data_matrix = np.zeros([pod, pod])
    for i in range(len(job_matrix)):
        sum_data_matrix += job_matrix[i]
    job_matrix_edit = copy.deepcopy(job_matrix)
    data_matrix_stuff, _ = bvn.solve_target_matrix(sum_data_matrix, pod)
    bvn_compose, _ = bvn.matrix_decompose(sum_data_matrix, data_matrix_stuff, pod, 0.8, num_bvn)
    sum_bvn = np.zeros([pod, pod])
    for i in range(len(bvn_compose)):
        sum_bvn += bvn_compose[i]
    if not is_strongly_connected_scipy(sum_bvn):
        last_compose = bvn_compose[-1]
        ring_compose = sub_ring_edit(last_compose, pod)
        bvn_compose[-1] = ring_compose
    link_matrix = port_allocate(bvn_compose, port, pod)
    bool_sum_data = np.where(sum_data_matrix > 0, 1, 0)
    bool_link = np.where(link_matrix > 0, 1, 0)
    t_ideal_low = np.sum(data_matrix_stuff) / (port * pod * band_per_port)
    flow_no_link = bool_sum_data - bool_link
    no_link_index_row, no_link_index_col = np.where(flow_no_link == 1)
    ideal_data_matrix = t_ideal_low * link_matrix * band_per_port
    delta_compose = ideal_data_matrix - sum_data_matrix
    stuff_compose = np.where(bool_link == 1, delta_compose, - np.inf)
    sum_data_edit = copy.deepcopy(sum_data_matrix)
    job_traffic_sort = np.argsort(-1 * np.array(single_traffic))
    for i in range(len(job_matrix)):
        job_index = job_traffic_sort[i]
        for j in range(len(no_link_index_row)):
            row, col = no_link_index_row[j], no_link_index_col[j]
            if job_matrix[job_index][row][col] == 0:
                continue
            else:
                _, path = max_min_weight_path_dijkstra_no_cycle(stuff_compose, row, col)
                for k in range(len(path) - 1):
                    job_matrix_edit[job_index][path[k]][path[k + 1]] += job_matrix[job_index][row][col] + 0
                    sum_data_edit[path[k]][path[k + 1]] += job_matrix[job_index][row][col]
                    stuff_compose[path[k]][path[k + 1]] -= job_matrix[job_index][row][col]
                job_matrix_edit[job_index][row][col] = 0
                sum_data_edit[row][col] -= job_matrix[job_index][row][col]
    link_row, link_col = np.where(link_matrix > 0)
    transmission_times = np.zeros([pod, pod])
    for i in range(len(link_row)):
        transmission_times[link_row[i]][link_col[i]] = (sum_data_edit[link_row[i]][link_col[i]] /
                                                        (band_per_port * link_matrix[link_row[i]][link_col[i]]))
    t_ideal_high = np.max(transmission_times)  # 取最慢的链路
    job_matrix_output, sum_matrix_output = binary_search(t_ideal_low, t_ideal_high, job_matrix, sum_data_matrix,
                                                         link_matrix, band_per_port, single_traffic, 1e-5,
                                                         job_link_index)
    return job_matrix_output, sum_matrix_output, link_matrix


# 分组方案
def transport_time(data, link_matrix, band_per_port):
    """

    :param band_per_port:
    :param data:
    :param link_matrix:
    :return:
    """
    flow_row, flow_col = np.where(link_matrix > 0)
    t = 0
    for i in range(len(flow_row)):
        row, col = flow_row[i], flow_col[i]
        t_link = data[row][col] / (link_matrix[row][col] * band_per_port)
        if t_link > t:
            t = t_link
    return t


def iteration_time(job_matrix, link_matrix_all, pod, band_per_port, long_group, short_group, long_train, short_train):
    """

    :param job_matrix:
    :param link_matrix_all:
    :param pod:
    :param band_per_port:
    :param long_group:
    :param short_group:
    :param long_train:
    :param short_train:
    :return:
    """
    data_long = np.zeros([pod, pod])
    data_short = np.zeros([pod, pod])
    for i in long_group:
        data_long += job_matrix[i]
    for i in short_group:
        data_short += job_matrix[i]
    long_flow = transport_time(data_long, link_matrix_all, band_per_port)
    short_flow = transport_time(data_short, link_matrix_all, band_per_port)
    long_bool = 0
    short_bool = 0
    if short_train > long_flow:
        short_bool = 1
    if long_train > short_flow:
        long_bool = 1
    return max(short_train, long_flow), max(long_train, short_flow), long_bool, short_bool


def match_degree_count(job_matrix, long_group, short_group, reverse_group, link_matrix_match, pod, band_per_port):
    """

    :param band_per_port:
    :param pod:
    :param job_matrix:
    :param long_group:
    :param short_group:
    :param reverse_group:
    :param link_matrix_match:
    :return:
    """
    data_long = np.zeros([pod, pod])
    data_short = np.zeros([pod, pod])
    for i in long_group:
        data_long += job_matrix[i]
    for i in short_group:
        data_short += job_matrix[i]
    long_flow = transport_time(data_long, link_matrix_match, band_per_port)
    short_flow = transport_time(data_short, link_matrix_match, band_per_port)
    match_degree_list_long = []
    match_degree_list_short = []
    for i in reverse_group:
        single_flow = transport_time(job_matrix[i], link_matrix_match, band_per_port)
        data_long_add = data_long + job_matrix[i]
        long_flow_add = transport_time(data_long_add, link_matrix_match, band_per_port)
        match_degree_long = (long_flow_add - long_flow) / single_flow
        data_short_add = data_short + job_matrix[i]
        short_flow_add = transport_time(data_short_add, link_matrix_match, band_per_port)
        match_degree_short = (short_flow_add - short_flow) / single_flow
        match_degree_list_long.append(match_degree_long)
        match_degree_list_short.append(match_degree_short)
    return match_degree_list_long, match_degree_list_short


def group(job_matrix, t_train, band_per_port, pod, link_matrix_group):
    """
    分组策略执行
    :param link_matrix_group:
    :param job_matrix:
    :param t_train:
    :param band_per_port:
    :param pod:
    :return:
    """
    train_sort_index = np.argsort(t_train)
    short_group = [train_sort_index[k] for k in range(len(job_matrix) - 1)]
    long_group = [train_sort_index[len(job_matrix) - 1]]
    t_train_long = t_train[train_sort_index[len(job_matrix) - 1]]
    flag = []
    t_group = []
    while 1:
        t_train_short = max([t_train[i] for i in short_group])
        t1, t2, long_bool, short_bool = iteration_time(job_matrix, link_matrix_group, pod, band_per_port, long_group,
                                                       short_group, t_train_long, t_train_short)
        if long_bool == 1 and short_bool == 0:  # flag = 2
            flag.append(2)
            t_group.append(t1 + t2)
            break
        elif long_bool == 1 and short_bool == 1:  # flag = 3
            flag.append(3)
            t_group.append(t1 + t2)
        elif long_bool == 0 and short_bool == 0:  # flag = 0
            flag.append(0)
            t_group.append(t1 + t2)
        elif long_bool == 0 and short_bool == 1:  # flag = 1
            flag.append(1)
            t_group.append(t1 + t2)
        long_group += [short_group[-1]]
        short_group = [train_sort_index[k] for k in range(len(short_group) - 1)]
    if flag[0] == 2:
        return short_group, long_group
    elif flag[0] == 3:
        if t_group[-1] <= t_group[-2]:
            short_group = [train_sort_index[k] for k in range(len(flag))]
            long_group = [train_sort_index[k] for k in range(len(job_matrix)) if train_sort_index[k] not in short_group]
            return short_group, long_group
        else:
            short_group = [train_sort_index[k] for k in range(len(flag) - 1)]
            long_group = [train_sort_index[k] for k in range(len(job_matrix)) if train_sort_index[k] not in short_group]
            return short_group, long_group
    else:
        len_flag = len(flag)  # flag 长度对应长组长度
        long_group = [train_sort_index[k] for k in range(len(job_matrix) - len_flag + 1, len(job_matrix))]
        t_train_long = train_sort_index[-1]
        short_group = [train_sort_index[len(job_matrix) - len_flag]]
        t_train_short = train_time[short_group[0]]
        reverse_group = [train_sort_index[k] for k in range(len(job_matrix) - len_flag)]
        while 1:
            if len(reverse_group) == 0:
                break
            data_long = np.zeros([pod, pod])
            data_short = np.zeros([pod, pod])
            for i in long_group:
                data_long += job_matrix[i]
            for i in short_group:
                data_short += job_matrix[i]
            long_flow = transport_time(data_long, link_matrix_group, band_per_port)
            short_flow = transport_time(data_short, link_matrix_group, band_per_port)
            match_long, match_short = (
                match_degree_count(job_matrix, long_group, short_group, reverse_group, link_matrix_group, pod,
                                   band_per_port))
            if long_flow >= t_train_short and short_flow < t_train_long:
                job_index = reverse_group[np.argmin(np.array(match_short))]
                short_group.append(job_index)
            elif long_flow < t_train_short and short_flow >= t_train_long:
                job_index = reverse_group[np.argmin(np.array(match_long))]
                long_group.append(job_index)
            else:
                if np.min(np.array(match_short)) < np.min(np.array(match_long)):
                    job_index = reverse_group[np.argmin(np.array(match_short))]
                    short_group.append(job_index)
                else:
                    job_index = reverse_group[np.argmin(np.array(match_long))]
                    long_group.append(job_index)
            reverse_group = [i for i in reverse_group if i != job_index]
        return short_group, long_group


# 主函数

job_number = 500
job1 = Schedule_part.generate_job(job_number)
all_job_index = [job1[i][0] for i in range(0, len(job1))]
single_link_out, sum_traffic_out = Schedule_part.traffic_count(job1)
usage = 0.4
iter_num = 10
flop = 275
train_time = Schedule_part.job_set_train(job1, flop, usage)
pod_number = 64
b_link = 30
port_num = 8
solution_out, undeploy_out, fix_job, unfix_job = Schedule_part.deploy_server(all_job_index, job1, pod_number, 512, 4)
f_job = [i for i in range(len(fix_job)) if fix_job[i] == 1]
uf_job = [i for i in range(len(fix_job)) if fix_job[i] == 0]
time1 = time.time()
data_matrix, link_job = job_ring(f_job, uf_job, solution_out, single_link_out, sum_traffic_out, pod_number)
time2 = time.time()
print(time2 - time1)
data_matrix, _, link_matrix_end = tpe(data_matrix, link_job, port_num, pod_number, 4, b_link, single_link_out)
time3 = time.time()
print(time3 - time2)
g1, g2 = group(data_matrix, train_time, b_link, pod_number, link_matrix_end)
time4 = time.time()
print(g1, g2)
print(time4 - time3)
train_g1 = [train_time[i] for i in g1]
train_g2 = [train_time[i] for i in g2]
t_iter = iteration_time(data_matrix, link_matrix_end, pod_number, b_link, g1, g2, max(train_g1), max(train_g2))
print(t_iter)
