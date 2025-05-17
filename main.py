import numpy as np
import bvn
import Schedule_part
from scipy.optimize import linear_sum_assignment


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
                worker_matrix[u][v] = sum_bvn[worker[u]][worker[v]]
        row_ind, col_ind = linear_sum_assignment(worker_matrix, maximize=True)
        data_matrix_single = np.zeros([len(row_ind), len(col_ind)])
        for u in range(len(row_ind)):
            data_matrix_single[worker[row_ind[u]]][worker[col_ind[u]]] = single_traffic[job_index]
        data_matrix_job[job_index] = data_matrix_single
        data_matrix_all += data_matrix_job[job_index]
    return data_matrix_job


# TPE 方案
def transmit():


def tpe(sum_data_matrix, port, pod):
    """
    tpe 策略
    :param pod: pod 数目
    :param sum_data_matrix: 总流量矩阵
    :param port: port 数目
    :return:
    """
    data_matrix_stuff, _ = bvn.solve_target_matrix(sum_data_matrix, pod)
    bvn_compose, bvn_sum = bvn.matrix_decompose(sum_data_matrix, data_matrix_stuff, pod, 0.8, - 1)
    reserve_compose = sum_data_matrix - bvn_sum

# 分组方案


# 主函数

job_number = 10
job1 = Schedule_part.generate_job(job_number)
all_job_index = [job1[i][0] for i in range(0, len(job1))]
single_link_out, sum_traffic_out = Schedule_part.traffic_count(job1)
usage = 0.4
iter_num = 10
flop = 275
train_time = Schedule_part.job_set_train(job1, flop, usage)
pod_number = 4
b_link = 30
port_num = 8
t_recon = 0.1
solution_out, undeploy_out, fix_job, unfix_job = Schedule_part.deploy_server(all_job_index, job1, pod_number, 512, 4)
f_job = [i for i in range(len(fix_job)) if fix_job[i] == 1]
uf_job = [i for i in range(len(fix_job)) if fix_job[i] == 0]
data_matrix = job_ring(f_job, uf_job, solution_out, single_link_out, sum_traffic_out, pod_number)
print(data_matrix)
