import numpy as np

#Косинусное сходство — это мера сходства между двумя векторами: в основном оно измеряет угол
#между ними и возвращает -1, если они прямо противоположны, и 1, если они абсолютно одинаковы.
#Важно отметить, что это мера ориентации, а не величины.

def cosine_distance(pose1, pose2):
    
    cossin = pose1.dot(np.transpose(pose2)) / (
        np.linalg.norm(pose1, axis=1) * np.linalg.norm(pose2, axis=1)
    )
    dist = np.diagonal(cossin).mean()

    return dist

