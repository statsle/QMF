# QMF
Quadratic Matrix Factorization Algorithm Implmented in Matlab

## Description
This code implements the Quadratic Matrix Factorization Algorithm proposed by [(Zheng Zhai, Hengchao Chen, Qiang Sun (2024))](https://arxiv.org/abs/2301.12965#:~:text=Quadratic%20Matrix%20Factorization%20with%20Applications%20to%20Manifold%20Learning,-Zheng%20Zhai%2C%20Hengchao&text=Matrix%20factorization%20is%20a%20popular,on%20which%20the%20dataset%20lies.). Firstly, Quadratic matrix factorization learns the curvature information contained in the noisy data. We use a quadratic function $f(\tau)$ to represent the fitting result. Secondly, by projection the data onto $f(\tau)$ via minimizing the Euclidean distance, we can obtain a new representation by denoising the original input.


##Functions
As the quadratic matrix factorization model without a regularization term can be easily realized by solving a regularized QMF with the regularized parameter lambda set to 0, we only need to explain how to use the RQMF function. From the perspective of how we select the local region, there are two versions of RQMF: QMF-local and QMF-kernel, which treat the constraint on the local region in two different ways. Regarding how we tune the regularization parameters, there are also two versions of QMF. RQMF-lambda directly solves the RQMF (regularized quadratic matrix factorization) model given a fixed lambda, while RQMF-adaptive adaptively tunes lambda in the algorithm.

The truncated local version of RQMF fits the data in the local region by selecting the K nearest neighbors, which we implement with the function RQMF(X, Tau, rho, adaptive, W) by assigning X with the selected data and setting the weighted matrix as the identity matrix I.

The soft kernel version of RQMF fits the data in the local region by assigning different weights to the data points, often realized by a kernel function. We implement the kernel RQMF with function RQMF(X, Tau, rho, adaptive, W) by assigning X with all the data and setting the weighted matrix with a diagonal matrix where the i-th element represents the important weight corresponding to the i-th sample.

The fixed lambda version of RQMF can be implemented by setting adaptive=0, while the adaptive version of RQMF can be accomplished by setting adaptive=1, where lambda is tuned in the algorithm to ensure that s'(lambda) is a constant.



## Demo
