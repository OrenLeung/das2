# DAS 2 🚀 - A Distributed Data Parallelism Library for Tensorflow Keras

![GitHub issues](https://img.shields.io/github/issues/OrenLeung/das2)
![GitHub contributors](https://img.shields.io/github/contributors/OrenLeung/das2)
![GitHub last commit](https://img.shields.io/github/last-commit/OrenLeung/das2)

**NOTE BETA: This is for experimenting and learning purposes. Please use [tf.distribute.Strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy) for not-learning workload**


Listened to [Stay (with Justin Bieber)](https://open.spotify.com/track/5HCyWlXZPP0y6Gqq8TgA20?si=5afd0b939f934759) on Loop for the whole 24 hours

## About
Deep Learning Models are becoming more and more compute intensive. Scaling horizontally through data parallelism helps this. By distributing batches across multiple worker machines (with each machine progressing a mini batch), this allows for more and more data to be progressed in parallel in theory*.

## Learning
Throughout this overnight 24 hour, I learned a lot about GradientTape, Keras, Tensorflow internals. Furthermore, in types of architecture I learned a lot about data and model parallelism and different distribution strategies.

## Architecture
1. Controller initializes the weights and distributes the same weights across every worker machine.
2. MiniBatches is ran on each worker then their gradients are accumulated by controller and averaged out then sent back to their workers.
3. Step 2 will repeat over and over again till it reaches a decent test accuracy

![DAS2 Architecture](https://docs.google.com/drawings/d/e/2PACX-1vSYEeVWRp_A_7hOpKPJa9rbku4RVBsXcBMMDgUsBdKfiGwSJ5SPT8rLbrgOj2_oJnRQh2SeLJm3ndRI/pub?w=960&h=720)
## Get Started 🚀
`pip install das2`

### Example

1. On Controller Machine, Run `python controller_run.py IP_ADDRESS_1 IP_ADDRESS_2`
2. On Worker 1 Machine, Run `python worker_run.py PORT1`
3. On Worker 2 Machine, Run `python worker_run.py PORT2`

## TODO 🚀
- [ ] Asynchronously Call the Workers <- **THIS IS IMPORTANT LOL**
- [ ] Abstract the Fast API endpoints in an Object
- [ ] Switch Over to GRPC/explore other RPC frameworks
- [ ] Experiment with All Reduce Algorithm
- [ ] Benchmark the performance vs single machine and also vs tf.distribute.Strategy. 
- [ ] track networking/cpu bottlenecks

## Resources 🚀
- https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy
- https://www.tensorflow.org/api_docs/python/tf/GradientTape
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
- https://ruder.io/optimizing-gradient-descent/
- https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD
- https://www.tensorflow.org/api_docs/python/tf/Tensor
- https://towardsdatascience.com/visual-intuition-on-ring-allreduce-for-distributed-deep-learning-d1f34b4911da
