"""
This file is used to deploy servers of each job. When each job arrives, Stimulate.py will start this thread and input
parameter as DP unit and so on. This thread will return timestamp of ending deploying and server deployment
program, which will be used in other thread.
"""
import numpy as np


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


def single_deploy_main(reverse_server, dp_unit_num, dp_unit_server):
    """
    This function is used to deploy each job in first part without first job
    :return:
    """


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

    # first part: deploy main flow:
    # first job deploys

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
