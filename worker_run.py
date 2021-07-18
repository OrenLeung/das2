import tensorflow as tf
from das2 import Worker
from tensorflow.keras.utils import to_categorical
from fastapi import Request
from fastapi import FastAPI, Depends
import uvicorn
import sys
NUM_WORKER = 2


def getModel() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax'),
        ]
    )
    return model


def getLoss():
    return tf.keras.losses.CategoricalCrossentropy(from_logits=False)


def getOptimizer():
    return tf.keras.optimizers.SGD(learning_rate=0.01)


def getData():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    BATCH_SIZE = 128
    SHUFFLE_BUFFER_SIZE = 100

    train_dataset = train_dataset.shuffle(
        SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE//NUM_WORKER).repeat(10)
    test_dataset = test_dataset.batch(BATCH_SIZE//NUM_WORKER)

    return train_dataset, test_dataset


train_dataset, test_dataset = getData()

newWorker = Worker(getModel(),  getOptimizer(),
                   getLoss(), train_dataset, test_dataset)


app = FastAPI()


@app.get("/getWeight")
def getWeight() -> str:
    return newWorker.serialWeightsJson()


async def get_body(request: Request):
    # or request.json() if you are sure the body is coming in that way
    return await request.body()


@app.post("/applyGradient")
def applyGradient(body=Depends(get_body)):
    # print(body.decode("utf-8"))
    # print(model.trainable_weights[3])

    newWorker.applyGradientFromSerial(body.decode("utf-8"))


@app.post("/setWeight")
def setWeight(body=Depends(get_body)):
    # print(body.decode("utf-8"))
    newWorker.loadWeightsFromSerial(body.decode("utf-8"))


@app.get("/eval")
def eval():
    test_loss, test_accuracy = newWorker.evaluate()
    return {"test_loss": test_loss, "test_accuracy": test_accuracy}

@app.get("/runMiniBatch")
def runMiniBatch():
    gradObject = newWorker.runMiniBatch()
    return gradObject

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(sys.argv[1]))