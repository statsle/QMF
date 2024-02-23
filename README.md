# QMF
Quadratic Matrix Factorization Algorithm Implmented in Matlab

## Description
This code implements the Quadratic Matrix Factorization Algorithm proposed by [(Zheng Zhai, Hengchao Chen, Qiang Sun (2024))](https://arxiv.org/abs/2301.12965#:~:text=Quadratic%20Matrix%20Factorization%20with%20Applications%20to%20Manifold%20Learning,-Zheng%20Zhai%2C%20Hengchao&text=Matrix%20factorization%20is%20a%20popular,on%20which%20the%20dataset%20lies.). Firstly, Quadratic matrix factorization learns the curvature information contained in the noisy data. We use a quadratic function $f(\tau)$ to represent the fitting result. Secondly, by projection the data onto $f(\tau)$ via minimizing the Euclidean distance, we can obtain a new representation by denoising the original input.


##Functions
From the perspective of how we select the local region, there are two versions of QMF: QMF-local and QMF-kernel, which treat the constraint onto the local region in two different ways.

QMF-local fits the data in the local region by selecting the K nearest neighbors, which we implement with the file QMF.m.

QMF-kernel fits the data in the local region by assigning the data points different weights, often realized by a kernel function. We implement QMF-kernel in the file QMF_K.m.

Regarding how we tune the regularization parameters, there are two versions of QMF. QMF-$\lambda$ directly solves the regularized RQMF (regularized quadratic matrix factorization) model given a fixed $\lambda$, while QMF-$adaptive$ adaptively tunes $\lambda$ in iterations.

## Demo
