"""
This file is used to deploy servers of each job. When each job arrives, Stimulate.py will start this thread and input
parameter as DP unit and so on. This thread will return timestamp of ending deploying and server deployment
program, which will be used in other thread.
"""
import numpy as np
import copy


def flow_split(percent, metric):
    """
    This function will divide init jobs into two part
    :param percent: split rate
    :param metric: traffic size
    :return: main flow and other flow of jobs, store in a queue, from larger job to tiner job
    """
    metric_np = np.array([metric[i] for i in range(0, len(metric))])
    sum_metric = np.sum(metric_np)
    main_flow = []
    min_flow = []
    len_of_metric = len(metric_np) + 0
    main_flow_size = 0
    for i in range(0, len_of_metric):
        max_metric_index = np.argmax(metric_np)
        if main_flow_size <= percent * sum_metric:
            main_flow.append(max_metric_index)
            main_flow_size += metric_np[max_metric_index]
            metric_np[max_metric_index] = 0
            continue
        else:
            min_flow.append(max_metric_index)
            metric_np[max_metric_index] = 0
    return main_flow, min_flow


def pre_ring_degree(rep_link):
    """
    This function is used to make degree of each node in topology not greater than 2
    :param rep_link: repeat link matrix.
    :return:
    """
    degree = np.sum(rep_link, axis=1)
    while 1:
        index = np.argmax(degree)
        if degree[index] <= 2:
            break
        else:
            relate_node = [k for k in range(0, len(degree)) if rep_link[index][k] == 1]
            degree_relate_node = np.array([degree[k] for k in relate_node])
            max_index = np.argmax(degree_relate_node)
            rep_link[index][relate_node[max_index]] = 0
            rep_link[relate_node[max_index]][index] = 0
            degree[index] -= 1
            degree[relate_node[max_index]] -= 1
    return rep_link


def find_sub_ring(matrix):
    """
    This function is used to find all rings in topo
    :param matrix: a topology may have some sub rings, while degree of each node in matrix is not bigger than 2
    :return:
    """
    count_matrix = copy.deepcopy(matrix)
    ring = []
    test_ring = []
    degree = np.sum(count_matrix, axis=1)
    degree_two_node = [k for k in range(0, len(degree)) if degree[k] == 2]
    while 1:
        if len(degree_two_node) == 0:
            break
        else:
            first_node = degree_two_node[0]
            test_ring.append(first_node)
            degree_two_node = [k for k in degree_two_node if k != first_node]
            while 1:
                next_node = np.argmax(count_matrix[test_ring[-1]])
                count_matrix[next_node][test_ring[-1]] = 0
                count_matrix[test_ring[-1]][next_node] = 0
                if next_node == first_node:
                    ring.append(test_ring)
                    test_ring = []
                    break
                if next_node not in degree_two_node:
                    test_ring = []
                    break
                degree_two_node = [k for k in degree_two_node if k != next_node]
                test_ring.append(next_node)
    return ring


def ring(matrix):
    """
    This function is used to format a ring without sub ring
    :param matrix:
    :return:
    """
    matrix = pre_ring_degree(matrix)
    sub_ring = find_sub_ring(matrix)
    all_ring = []
    for i in range(0, len(sub_ring)):
        all_ring += sub_ring
    return all_ring


def server_local(reverse_server, dp_unit_num, dp_unit_server)


def single_deploy_main(reverse_server, dp_unit_num, dp_unit_server, link_matrix):
    """
    This function is used to deploy each job in first part without first job
    :param reverse_server: server pool
    :param dp_unit_num: amount of DP unit of this job
    :param dp_unit_server: amount of server of each DP unit of this job
    :param link_matrix: a matrix stores links should exist between each ToR pair
    :return:
    """
    local_server = []
    traffic_metric = np.zeros(len(reverse_server))
    for i in range(0, dp_unit_num):
        vacant_index = np.argmax(reverse_server)
        if dp_unit_server <= reverse_server[vacant_index]:
            local_server.append([(vacant_index, dp_unit_server)])
            reverse_server[vacant_index] -= dp_unit_server
        else:
            count_unit = dp_unit_server
            local_server.append([])
            while count_unit > 0:
                vacant_index = np.argmax(reverse_server)
                local_server[-1].append((vacant_index, reverse_server[vacant_index]))
                count_unit -= reverse_server[vacant_index]
                reverse_server[vacant_index] = 0
    ring_matrix = np.zeros([len(reverse_server), len(reverse_server)], dtype=bool)
    moe_matrix = np.zeros([len(reverse_server), len(reverse_server)], dtype=bool)
    all_reduce_tor = []
    for i in range(0, len(local_server)):
        if len(local_server[i]) > 1:
            rep_link = np.zeros([len(reverse_server), len(reverse_server)], dtype=bool)
            moe_dp = np.zeros(len(local_server[i]))
            for j in range(0, moe_dp):
                moe_dp[j] = local_server[i][j][0]
            for p in moe_dp:
                for q in moe_dp:
                    if link_matrix[p][q] == 1:
                        rep_link[p][q] = 1
            moe_ring = ring(rep_link)
            for j in range(0, len(moe_ring) - 1):
                moe_matrix[moe_ring[j]][moe_ring[j + 1]] = 1
            all_reduce_tor.append(moe_ring[0])
        else:
            all_reduce_tor.append(local_server[i][0][0])
    link_matrix += moe_matrix
    rep_link = np.zeros([len(reverse_server), len(reverse_server)], dtype=bool)
    for p in all_reduce_tor:
        for q in all_reduce_tor:
            if link_matrix[p][q] == 1:
                rep_link[p][q] = 1
    dp_ring = ring(rep_link)
    for i in range(0, len(dp_ring) - 1):
        ring_matrix[dp_ring[i]][dp_ring[i + 1]] = 1
    ring_matrix[dp_ring[-1]][0] = 1
    return ring_matrix, moe_matrix


def init_deploy(dp_unit_num_array, dp_unit_server_array, batch_size, reverse_server_pool):
    """
    This function is used to deploy the DP units of jobs before scheduling flows.
    :param dp_unit_num_array: an array stores the amount of DP units of all initial jobs
    :param dp_unit_server_array: an array stores the server amount of all initial jobs
    :param batch_size: an array stores the batch size of each job in an iteration
    :param reverse_server_pool: an array stores the amount of server each ToR has
    :return:
    """
    reverse_server_pool = np.array([reverse_server_pool[i] for i in range(0, len(reverse_server_pool))])
    dp_server_num = [dp_unit_server_array[i] * dp_unit_num_array[i] for i in range(0, len(dp_unit_num_array))]
    # an array store amount of server used to train each job
    traffic_metric = [dp_server_num[i] / batch_size[i] for i in range(0, len(dp_unit_num_array))]
    # an array store the traffic metric, which denotes the traffic size of each job
    if sum(dp_server_num) > sum(reverse_server_pool):
        return -1
    split_percent = 0.7
    # split rate, which is used to judge mainstream flow and tine flow
    main, vice = flow_split(split_percent, traffic_metric)
    # separate jobs with split rate
    all_reduce_matrix = [np.zeros([len(reverse_server_pool), len(reverse_server_pool)], dtype=bool)]
    moe_matrix = [np.zeros([len(reverse_server_pool), len(reverse_server_pool)], dtype=bool)]

    # first part: deploy main flow:
    # first job deploys

    local_info = []
    local_server = []
    for i in range(0, dp_unit_num_array[main[0]]):
        vacant_index = np.argmax(reverse_server_pool)
        # find the index of target ToR
        if dp_unit_server_array[main[0]] <= reverse_server_pool[vacant_index]:
            reverse_server_pool[vacant_index] -= dp_unit_server_array[main[0]]
            local_server.append([(vacant_index, dp_unit_server_array[main[0]])])
        else:
            count_unit = dp_unit_server_array[main[0]]
            local_server.append([])
            while count_unit > 0:
                vacant_index = np.argmax(reverse_server_pool)
                local_server[-1].append((vacant_index, reverse_server_pool[vacant_index]))
                count_unit -= reverse_server_pool[vacant_index]
                reverse_server_pool[vacant_index] = 0

    local_info.append(local_server)
    # put the information first job into local information, notice that the index of this job is not the first.
    dp_tor = [local_server[i][0][0] for i in range(0, len(local_server))]
    for i in range(0, len(dp_tor) - 1):
        all_reduce_matrix[0][dp_tor[i]][dp_tor[i + 1]] = 1
        all_reduce_matrix[0][dp_tor[i + 1]][dp_tor[i]] = 1
    all_reduce_matrix[0][dp_tor[0]][dp_tor[-1]] = 1
    all_reduce_matrix[0][dp_tor[-1]][dp_tor[0]] = 1
    for i in range(0, len(dp_tor)):
        if len(local_server[i]) > 1:
            dp_moe = [local_server[i][j][0] for j in range(0, len(local_server))]
            for j in range(0, len(dp_moe) - 1):
                moe_matrix[0][dp_moe[j]][dp_moe[j + 1]] = 1
                moe_matrix[0][dp_moe[j] + 1][dp_moe[j]] = 1
    # update link matrix

    # other job in main queue


def server_deploy(som, lm, tm, dp_unit_num, dp_unit_server, traffic_size_array):
    """
    This function is used to deploy the DP units of a job when scheduling flows.
    :param som: server occupation matrix (1 × Pod number), each element signify the amount of server used on rach pod
    :param lm: link matrix (Pod number × Pod number), each element signify the amount of oxc links between this Pod Pair
    :param tm: traffic matrix(Pod number × Pod number), each element signify the size of traffic between this Pod Pair
    :param dp_unit_num: the amount of DP unit of this job
    :param dp_unit_server: the server amount of each DP unit of this job
    :param traffic_size_array: an array stores traffic matrix of all jobs
    :return:
    """


A = np.array([[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 1],
              [0, 0, 0, 1, 1, 0]])
print(find_sub_ring(A))
