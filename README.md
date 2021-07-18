# DAS 2 🚀 - A Distributed Data Parallelism Library for Tensorflow Keras

**NOTE BETA: This is for experimenting and learning purposes. Please use [tf.distribute.Strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy) for not-learning workload**


Listened to [Stay (with Justin Bieber)](https://open.spotify.com/track/5HCyWlXZPP0y6Gqq8TgA20?si=5afd0b939f934759) on Loop for the whole 24 hours

## About
Deep Learning Models are becoming more and more compute intensive. Scaling horizontally through data parallelism helps this. By distributing batches across multiple worker machines (with each machine progressing a mini batch), this allows for more and more data to be progressed in parallel in theory*.
## Get Started 🚀
`pip install das2`

### Example

On Controller Machine, Run `python controller_run.py IP_ADDRESS_1 IP_ADDRESS_2`
On Worker 1 Machine, Run `python worker_run.py PORT1`
On Worker 2 Machine, Run `python worker_run.py PORT2`

## TODO 🚀
- [ ] Asynchronously Call the Workers <- **THIS IS IMPORTANT LOL**
- [ ] Abstract the Fast API endpoints in an Object
- [ ] Switch Over to GRPC/explore other RPC frameworks
- [ ] Experiment with All Reduce Algorithm
- [ ] Benchmark the performance vs single machine and also vs tf.distribute.Strategy. 
- [ ] track networking/cpu bottlenecks

## Resources 🚀
- https://towardsdatascience.com/visual-intuition-on-ring-allreduce-for-distributed-deep-learning-d1f34b4911da
