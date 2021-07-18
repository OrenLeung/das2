import numpy as np
import json
import requests
from typing import List


def serialGradient(gradient):
    gradList = []
    for grad in gradient:
        # print(grad.shape)
        gradList.append(grad.tolist())
    gradObject = {"tensor": gradList}
    # gradJson = json.dumps(gradObject)

    return gradObject


def deSerialWeightsJson(serial) -> List[np.ndarray]:
    finalWeightList = []
    weightObject = json.loads(serial)
    for weights in weightObject["tensor"]:
        finalWeightList.append(np.array(weights, dtype="float32"))
    return finalWeightList


def averageWeights(weightSets: List[List[np.ndarray]]) -> List[np.ndarray]:
    averageWeightsList: List[np.ndarray] = []
    for i in range(len(weightSets[0])):
        newWeightList = []
        for j in range(len(weightSets)):
            # print(weightSets[j][i].shape)
            newWeightList.append(weightSets[j][i])
        npWeightAverage = (np.array(newWeightList)).mean(axis=0)
        averageWeightsList.append(npWeightAverage)

    return averageWeightsList


class Controller:
    workers_ip: List[str]

    def __init__(self, workers_ip: List[str]):
        self.workers_ip = workers_ip

        # get intialWeights
        initalWeights = requests.get(f"{workers_ip[0]}/getWeight")

        for worker_ip in workers_ip:
            requests.post(f"{worker_ip}/setWeight",
                          data=initalWeights.json())

    def run(self, numOfSteps):
        for i in range(numOfSteps):
            gradList = []

            for worker_ip in self.workers_ip:
                grad = deSerialWeightsJson(requests.get(
                    f"{worker_ip}/runMiniBatch").text)
                gradList.append(grad)

            averagegradsList = averageWeights(gradList)

            for worker_ip in self.workers_ip:
                _ = requests.post(f"{worker_ip}/applyGradient",
                                  json=serialGradient(averagegradsList))

            if i % 100 == 0:
                for worker_ip in self.workers_ip:
                    test_results = requests.get(
                        f"{worker_ip}/eval").json()
                    print(test_results)
