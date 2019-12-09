# DGS PyTorch
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Modern large scale machine learning applications require stochastic optimization algorithms to be implemented with distributed computational architectures. A key bottleneck is the communication overhead for exchanging information, such as stochastic gradients, among different nodes. Recently, gradient sparsification techniques have been proposed to reduce communications cost and thus alleviate the network overhead. However, most of gradient sparsification techniques consider only synchronous parallelism and cannot be applied in asynchronous distributed training.

In this project, we present a dual-way gradient sparsification approach (DGS) that is suitable for asynchronous distributed training.

![Architecture of PS Based ASGD and DGS](https://yanring-1252048839.cos.ap-guangzhou.myqcloud.com/img/20191209143252.png)

We implemented a async parameter server based on PyTorch gloo backend. Our optimizer implemented 5 training methonds: DGS, ASGD, GradientDropping, DeepGradientCompression, single node momentum SGD. 

## Features
1. Async parameter server on PyTorch
2. Support gradient sparsiﬁcation, e.g. gradient dropping [1].
3. Sparse communication.
4. GPU training.
5. DALI dataloader for imagenet.
6. Two distributed example script: 1. cifar10 2. ImageNet.

## Performance

Experiments with different scales on ImageNet and CIFAR-10 show that:   
(1) compared with ASGD, Gradient Dropping and Deep Gradient Compression, DGS with SAMomentum consistently achieves  better performance;   
(2) DGS improves the scalability of asynchronous training, especially with limited networking infrastructure.

![Speedups for DGS and ASGD with 10Gbps and 1Gbps Ethenet](https://yanring-1252048839.cos.ap-guangzhou.myqcloud.com/img/20191209143644.png)

Figure above shows the training speedup with different network bandwidth values.
As the number of workers increases, the acceleration of ASGD decreases to nearly zero due to the bottleneck of communication. In contrast, DGS achieves nearly linear speedup with 10Gbps. With 1Gbps network, ASGD only achieves $1\times$ speedup with 16 workers, while DGS achieves $12.6\times$ speedup, which proves the the superiority of our DGS under low bandwidth.


## Quick Start
```
# environ setup

git clone https://github.com/yanring/GradientServer.git

cd GradientServer

pip install -r requirements.txt
```

```
# distributed cifar-10 example
# On the server (rank 0 is the server)
python example/cifar10.py --world-size 2 --rank 0 --cuda
# On the worker
python example/cifar10.py --world-size 2 --rank 1 --cuda
```


## Use DGS in Your Code

First, you should start a server.
```
init_server(args, net)
```
Second, replace your torch.optimizer.sgd with GradientSGD.
```

optimizer = GradientSGD(net.parameters(), lr=args.lr, model=net, momentum=args.momentum,weight_decay=args.weight_decay,args=args)
```

## Limitations and Future Plans
TODO

## Publications
TODO

## Refrence
[1] Alham Fikri Aji and Kenneth Heaﬁeld, ‘Sparse communication for distributed gradient descent’, arXiv preprint arXiv:1704.05021, (2017).
