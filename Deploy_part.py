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
    output_ring = []
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
                    output_ring.append(test_ring)
                    test_ring = []
                    break
                if next_node not in degree_two_node:
                    test_ring = []
                    break
                degree_two_node = [k for k in degree_two_node if k != next_node]
                test_ring.append(next_node)
    return output_ring


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


def single_deploy_main(reverse_server, dp_unit_num, dp_unit_server, link_matrix, tor_traffic, traffic_metric):
    """
    This function is used to deploy each job in first part
    :param reverse_server: server pool
    :param dp_unit_num: amount of DP unit of this job
    :param dp_unit_server: amount of server of each DP unit of this job
    :param link_matrix: a matrix stores links should exist between each ToR pair
    :param tor_traffic: output traffic metric of each ToR
    :param traffic_metric: traffic metric of this job's each DP unit
    :return:
    """
    local_server = []
    for i in range(0, dp_unit_num):
        vacant_index = np.argmin(tor_traffic)
        if dp_unit_server <= reverse_server[vacant_index]:
            local_server.append([(vacant_index, dp_unit_server)])
            reverse_server[vacant_index] -= dp_unit_server
            tor_traffic[vacant_index] += traffic_metric
        else:
            count_unit = dp_unit_server
            local_server.append([])
            while count_unit > 0:
                test_tor_traffic = copy.deepcopy(tor_traffic)
                max_traffic = np.max(test_tor_traffic)
                for j in range(0, len(tor_traffic)):
                    if reverse_server[j] == 0:
                        test_tor_traffic[j] = max_traffic + 10
                vacant_index = np.argmin(test_tor_traffic)

                tor_deploy = min(count_unit, reverse_server[vacant_index])
                local_server[-1].append((vacant_index, tor_deploy))
                count_unit -= tor_deploy
                tor_traffic[vacant_index] += traffic_metric * tor_deploy / dp_unit_server
                reverse_server[vacant_index] -= tor_deploy
    ring_matrix = np.zeros([len(reverse_server), len(reverse_server)], dtype=int)
    moe_matrix = np.zeros([len(reverse_server), len(reverse_server)], dtype=int)
    all_reduce_tor = []
    for i in range(0, len(local_server)):
        if len(local_server[i]) > 1:
            moe_dp = [local_server[i][j][0] for j in range(0, len(local_server[i]))]
            for j in range(1, len(moe_dp)):
                moe_matrix[moe_dp[0]][moe_dp[j]] = 1
                moe_matrix[moe_dp[j]][moe_dp[0]] = 1
        all_reduce_tor.append(local_server[i][0][0])
    rep_link = np.zeros([len(reverse_server), len(reverse_server)], dtype=int)
    for i in all_reduce_tor:
        for j in all_reduce_tor:
            if link_matrix[i][j] == 1:
                rep_link[i][j] = 1
    ring_topo = ring(rep_link)
    for i in range(0, len(all_reduce_tor)):
        if all_reduce_tor[i] not in ring_topo:
            ring_topo.append(all_reduce_tor[i])
    for i in range(0, len(ring_topo) - 1):
        ring_matrix[ring_topo[i]][ring_topo[i + 1]] = 1
        ring_matrix[ring_topo[i + 1]][ring_topo[i]] = 1
    ring_matrix[ring_topo[0]][ring_topo[-1]] = 1
    ring_matrix[ring_topo[-1]][ring_topo[0]] = 1
    return ring_matrix, moe_matrix, local_server


def var_count(ring_matrix, dp_unit_server, tor_traffic_metric, reverse_server, job_metric):
    """
    This function is used to count zhe var of one choice of ring topo
    :param ring_matrix:
    :param dp_unit_server:
    :param tor_traffic_metric:
    :param reverse_server:
    :param job_metric:
    :return:
    """
    local_server = []
    sum_ring = np.sum(ring_matrix, axis=1)
    ring_topo_choose = []
    for i in range(0, len(sum_ring)):
        if sum_ring[i] > 0:
            ring_topo_choose.append(i)
    reserve_unit = np.array([dp_unit_server for k in range(0, len(ring_topo_choose))])
    test_tor_traffic_metric = copy.deepcopy(tor_traffic_metric)
    for i in range(0, len(ring_topo_choose)):
        local_server.append([])
        tor_count = min(dp_unit_server, reverse_server[ring_topo_choose[i]])
        local_server[i].append((tor_count, ring_topo_choose[i]))
        reserve_unit[i] -= tor_count
        test_tor_traffic_metric[ring_topo_choose[i]] += job_metric * tor_count / dp_unit_server
    if np.sum(reserve_unit) <= 0:
        return np.var(test_tor_traffic_metric), test_tor_traffic_metric, local_server
    else:
        for i in range(0, len(ring_topo_choose)):
            while reserve_unit[i] > 0:
                test2 = copy.deepcopy(test_tor_traffic_metric)
                for j in range(0, len(test_tor_traffic_metric)):
                    if reverse_server[j] == 0:
                        test2[j] = np.max(test_tor_traffic_metric) + 10
                index = np.argmin(test2)
                tor_count = min(reverse_server[index], reserve_unit[i])
                local_server[i].append((tor_count, index))
                reserve_unit[i] -= tor_count
                test_tor_traffic_metric[index] += job_metric * tor_count / dp_unit_server
        return np.var(test_tor_traffic_metric), test_tor_traffic_metric, local_server


def single_deploy_vice(ring_topo, dp_unit_server, dp_unit_num, tor_traffic_metric, reverse_server, job_metric):
    """
    This function is used to deploy each job in second part
    :param ring_topo: ring topo set after first part
    :param dp_unit_server:
    :param dp_unit_num:
    :param tor_traffic_metric:
    :param reverse_server:
    :param job_metric:
    :return:
    """
    relate_ring = []
    for i in range(0, ring_topo):
        if np.sum(ring_topo[i]) == 2 * dp_unit_num:
            relate_ring.append(ring_topo[i])
    if len(relate_ring) == 0:
        return [], [], [], []
    relate_ring_var = []
    relate_ring_metric = []
    relate_ring_plan = []
    for i in relate_ring:
        relate_var, relate_metric, relate_local_server = var_count(i, dp_unit_server, tor_traffic_metric,
                                                                   reverse_server, job_metric)
        relate_ring_var.append(relate_var)
        relate_ring_metric.append(relate_metric)
        relate_ring_plan.append(relate_local_server)
    index = np.argmin(relate_ring_var)
    ring_matrix = np.zeros([len(reverse_server), len(reverse_server)])
    moe_matrix = np.zeros([len(reverse_server), len(reverse_server)])
    dp_tor = []
    for i in range(0, len(relate_ring_plan[index])):
        dp_tor.append(relate_ring_plan[index][i][0][0])
        if len(relate_ring_plan[index][i]) > 1:
            moe_tor = []
            for j in range(0, len(relate_ring_plan[index][i])):
                moe_tor.append(relate_ring_plan[index][i][j][0])
                reverse_server[relate_ring_plan[index][i][j][0]] -= reverse_server[relate_ring_plan[index][i][j][1]]
            for j in range(1, len(moe_tor)):
                moe_matrix[moe_tor[0]][moe_tor[j]] = 1
                moe_matrix[moe_tor[j]][moe_tor[0]] = 1
        reverse_server[relate_ring_plan[index][i][0][0]] -= reverse_server[relate_ring_plan[index][i][0][1]]
    for i in range(0, len(dp_tor) - 1):
        ring_matrix[dp_tor[i]][dp_tor[i + 1]] = 1
        ring_matrix[dp_tor[i + 1]][dp_tor[i]] = 1
    ring_matrix[dp_tor[0]][dp_tor[-1]] = 1
    ring_matrix[dp_tor[-1]][dp_tor[0]] = 1
    return relate_ring_plan[index], relate_ring_metric[index], ring_matrix, moe_matrix


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
    tor_traffic_metric = np.zeros(len(reverse_server_pool))
    # an array store the traffic metric, which denotes the traffic size of each job
    if sum(dp_server_num) > sum(reverse_server_pool):
        return -1
    split_percent = 0.7
    # split rate, which is used to judge mainstream flow and tine flow
    main, vice = flow_split(split_percent, traffic_metric)
    # separate jobs with split rate
    all_reduce_matrix = [np.zeros([len(reverse_server_pool), len(reverse_server_pool)], dtype=bool)]
    moe_matrix = [np.zeros([len(reverse_server_pool), len(reverse_server_pool)], dtype=bool)]
    link_m = np.zeros([len(reverse_server_pool), len(reverse_server_pool)])

    local_info = []
    # first part: deploy main flow:

    all_topo = []
    for i in range(0, len(main)):
        ring_topo, moe_topo, local_server = (
            single_deploy_main(reverse_server_pool, dp_unit_num_array[main[i]], dp_unit_server_array[main[i]], link_m,
                               tor_traffic_metric, traffic_metric[main[i]]))

        # update link matrix
        local_info.append(local_server)
        all_topo.append(ring_topo)
        for p in range(0, len(reverse_server_pool)):
            for q in range(0, len(reverse_server_pool)):
                if ring_topo[p][q] == 1 or moe_topo[p][q] == 1:
                    link_m[p][q] = 1

    # second part: deploy vice flow:
    for i in range(0, len(vice)):
        local_server, traffic_metric, ring_topo, moe_topo = (
            single_deploy_vice(all_topo, dp_unit_server_array[vice[i]], dp_unit_num_array[vice[i]],
                               tor_traffic_metric, reverse_server_pool, traffic_metric[vice[i]]))
        if len(local_server) == 0:
            ring_topo, moe_topo, local_server = (
                single_deploy_main(reverse_server_pool, dp_unit_num_array[main[i]], dp_unit_server_array[main[i]],
                                   link_m, tor_traffic_metric, traffic_metric[main[i]]))
            all_topo.append(ring_topo)
            local_info.append(local_server)
            for p in range(0, len(reverse_server_pool)):
                for q in range(0, len(reverse_server_pool)):
                    if ring_topo[p][q] == 1 or moe_topo[p][q] == 1:
                        link_m[p][q] = 1
        else:
            local_info.append(local_server)
    return local_info


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
