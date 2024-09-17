"""
This file is used to deploy servers of each job. When each job arrives, Stimulate.py will start this thread and input
parameter as DP unit and so on. This thread will return timestamp of ending deploying and server deployment
program, which will be used in other thread.
"""


def init_deploy(dp_unit_num_array, dp_unit_server_array, traffic_size_array):
    """
    This function is used to deploy the DP units of jobs before scheduling flows.
    :param dp_unit_num_array: an array stores the amount of DP units of all initial jobs
    :param dp_unit_server_array: an array stores the server amount of all initial jobs
    :param traffic_size_array: an array stores traffic matrix of all jobs
    :return:
    """


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

