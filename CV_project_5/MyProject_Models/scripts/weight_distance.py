import numpy as np
import math

#Взвешенное совпадение позволяет учесть степень достоверности ключевой точки: соединения с низкой
#достоверностью должны оказывать меньшее влияние на показатель совпадения, чем соединения с высокой
#степенью достоверности.

def weight_distance(pose1, pose2, conf1):
    # D(U,V) = (1 / sum(conf1)) * sum(conf1 * ||pose1 - pose2||) = sum1 * sum2

    sum1 = 1 / np.sum(conf1)
    sum2 = 0

    for i in range(len(pose1)):
        # каждый индекс i имеет x и y, у которых одинаковая оценка достоверности
        conf_ind = math.floor(i / 2)
        sum2 = conf1[conf_ind] * abs(pose1[i] - pose2[i])

    weighted_dist = sum1 * sum2

    return weighted_dist

