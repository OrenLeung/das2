from fastapi.applications import FastAPI
# import uvicorn
from typing import Any, Iterator, List, Tuple
import tensorflow as tf
import json
import numpy as np


class Worker():
    model: tf.keras.Model
    optimizer: tf.keras.optimizers.Optimizer
    loss_function: tf.keras.losses.Loss
    metrics: List[str]
    training_iterator: Iterator[Any]

    def __init__(self, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, loss_function: tf.keras.losses.Loss, training_data: tf.data.Dataset, eval_data=tf.data.Dataset, metrics: List[str] = ["accuracy"]):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.training_data = training_data.repeat(100)
        self.eval_data = eval_data
        self.training_iterator = iter(self.training_data)

        model.compile(optimizer=self.optimizer,
                      loss=self.loss_function,
                      metrics=self.metrics
                      )

    def serialWeightsJson(self) -> str:
        def tensorToList(tensor: tf.Tensor) -> List[Any]:
            return tensor.numpy().tolist()

        weightsListType = list(map(tensorToList, self.model.trainable_weights))

        weightObject = {"tensor": weightsListType}

        weightSerial: str = json.dumps(weightObject)

        return weightSerial

    def _deSerialTensorJson(self, serial: str) -> List[np.ndarray]:
        finalTensorList = []
        tensorObject = json.loads(serial)
        for tensor in tensorObject["tensor"]:
            finalTensorList.append(np.array(tensor, dtype="float32"))

        return finalTensorList

    def loadWeightsFromSerial(self, serial: str) -> None:
        newWeightList: List[np.ndarray] = self._deSerialTensorJson(serial)
        if len(newWeightList) == len(self.model.trainable_weights):
            for i, newWeight in enumerate(newWeightList):
                self.model.trainable_weights[i].assign(newWeight)
        else:
            raise Exception('loadWeight len !== model.trainable_weights')

    def applyGradientFromSerial(self, serial: str) -> None:
        gradient: List[np.ndarray] = self._deSerialTensorJson(serial)
        if len(gradient) == len(self.model.trainable_weights):
            # for i, newWeight in enumerate(newWeightList):
            #     model.trainable_weights[i].assign(newWeight)
            self.optimizer.apply_gradients(
                zip(gradient, self.model.trainable_weights))
        else:
            raise Exception('loadWeight len !== model.trainable_weights')

    def evaluate(self) -> Tuple[float, float]:
        test_loss, test_accuracy = self.model.evaluate(self.eval_data)
        return test_loss, test_accuracy

    def runMiniBatch(self) -> Any:

        x_true, y_true = self.training_iterator.get_next()  # type: ignore
        with tf.GradientTape() as tape:
            tape.reset()
            tape.watch(self.model.trainable_weights)
            output = self.model(x_true)
            loss = self.loss_function(y_true, output)
            print(loss)
        gradient = tape.gradient(loss, self.model.trainable_weights)

        def tensorToList(tensor: tf.Tensor) -> List[Any]:
            return tensor.numpy().tolist()

        gradList = list(map(tensorToList, gradient))

        gradObject = {"tensor": gradList}

        return gradObject
