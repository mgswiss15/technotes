\(
\def\rv#1{{\mathbf #1}}
\def\pt{p_\theta}
\def\rvx{\rv x}
\def\E{\mathbb{E}}
\def\gN{\mathcal{N}}
\)
# Evaluation of diffusion model via importance sampling of NLL

***Created: 2024-01-21, 
Updated: 2024-01-26, 
Updated: 2024-01-28***

This uses the standard DDPM [^1] notation, the importance sampling was inspired by Importance Weighted VAE [^2].

## Importance sampling estimation of data probability
We evaluate the fidelity of data samples via the marginal negative log-likelihood (NLL) estimated by importance sampling \cite{burda2015importance, tomczak2018vae}. 

We start from the model marginal
$$
\begin{aligned}
\pt(\rvx_0)
=  & \int \pt(\rvx_0 | \rvx_{1:T}) \pt(\rvx_{1:T}) \, d\rvx_{1:T} \\
=  & \int q(\rvx_{1:T} | \rvx_0) \frac{\pt(\rvx_0 | \rvx_{1:T}) \pt(\rvx_{1:T})}{q(\rvx_{1:T} | \rvx_0)} \, d\rvx_{1:T} \enspace ,
\end{aligned}
$$
and its importance weighted monte-carlo (MC) estimate based on $M$ samples
$$
\pt(\rvx_0)
\approx
\frac{1}{M}\sum_{i=1}^M \frac{\pt(\rvx_{1:T}^{(i)}) \pt(\rvx_0 | \rvx_{1:T}^{(i)})}{q(\rvx_{1:T}^{(i)} | \rvx_0)},
\quad \rvx_{t}^{(i)} \sim q(\rvx_{t} | \rvx_0) = \gN \left(\rvx_t; \sqrt{\bar{a}_t}\rvx_{0}, (1 - \bar{a}_t) \rv{I}\right), \ \forall t=1, \ldots, T \enspace ,
$$
where we used the identity $q(\rvx_{1:T} | \rvx_0) = \prod_{t=1}^T q(\rvx_{t} | \rvx_0)$.

From the standard DDPM assumptions and their results for the forward process
$$
q(\rvx_t | \rvx_{t-1}) = q(\rvx_t | \rvx_{t-1}, \rvx_0) = \frac{q(\rvx_{t-1} | \rvx_{t}, \rvx_0) q(\rvx_{t} | \rvx_0) q(\rvx_0)}{q(\rvx_{t-1} | \rvx_0) q(\rvx_0)}
$$
and hence
$$
q(\rvx_{1:T} | \rvx_0) = \prod_{t=1}^T q(\rvx_t | \rvx_{t-1}) = q(\rvx_1 | \rvx_0) \prod_{t=2}^T \frac{q(\rvx_{t-1} | \rvx_{t}, \rvx_0) q(\rvx_{t} | \rvx_0)}{q(\rvx_{t-1} | \rvx_0)}
= q(\rvx_1 | \rvx_0) \frac{q(\rvx_{T} | \rvx_0)}{q(\rvx_1 | \rvx_0)} \prod_{t=2}^T q(\rvx_{t-1} | \rvx_{t}, \rvx_0)
= q(\rvx_{T} | \rvx_0) \prod_{t=2}^T q(\rvx_{t-1} | \rvx_{t}, \rvx_0) \enspace .
$$
Similarly for the reverse process we get
\begin{equation}
\pt(\rvx_0 | \rvx_{1:T}) \pt(\rvx_{1:T})  = 
\pt(\rvx_T) \pt(\rvx_0 | \rvx_1) \prod_{t=2}^T \pt(\rvx_{t-1} | \rvx_t) \enspace .
\end{equation}

Bringing these together the importance sampling estimate is
$$\pt(\rvx_0) \approx  
\frac{1}{M}\sum_{i=1}^M \pt(\rvx_{0} | \rvx_{1}^{(i)}) \frac{\pt(\rvx_T^{(i)})}{q(\rvx_{T}^{(i)} | \rvx_0)}
\prod_{t=2}^T \frac{\pt(\rvx_{t-1}^{(i)} | \rvx_{t})}{q(\rvx_{t-1}^{(i)} | \rvx_{t}^{(i)}, \rvx_0)}
\enspace ,
\tag{1}
\label{eq:mc-estimate}
$$

### Stable evaluation

Obviously, taking products of distributions is extremely unstable and hence we are better of working in $\log$s.
$$\pt(\rvx_0) \approx
\frac{1}{M}\sum_{i=1}^M \exp \left( \underbrace{\log \pt(\rvx_{0} | \rvx_{1}^{(i)})}_\text{data conditional} + \underbrace{\log \pt(\rvx_T^{(i)}) - \log q(\rvx_{T}^{(i)} | \rvx_0)}_\text{prior matching} + \sum_{t=2}^T \left[ \underbrace{\log \pt(\rvx_{t-1}^{(i)} | \rvx_{t}^{(i)}) - \log q(\rvx_{t-1}^{(i)} | \rvx_{t}^{(i)}, \rvx_0)}_\text{denoising matching} \right] \right)
\enspace .
\tag{2}
\label{eq:mc-stable}
$$

### Per-pixel results

It is conventional to report results per-pixel rather than per the whole image. This explains how to get those. 

For a D-variate gaussian pdf with diagonal covariance $f(\rvx) = \gN(\rvx; \rv{\mu}, \sigma^2 \rv{I})$ we have $f(\rvx) = \prod_{i=1}^D f_i(x_i)$, where $f_i(x_i) = \gN(x_i; \mu_i, \sigma^2)$.
We can therefore evaluate all log-probabilities in \eqref{eq:mc-stable} as sum of elementwise (pixel) log probabilities so for example
$$
\log \pt(\rvx_{0} | \rvx_{1}^{(i)}) = \log \prod_{j=1}^D p_j(x_{0j} | x_{1j}^{(i)}) = \sum_{j=1}^D \log p_j(x_{0j} | x_{1j}^{(i)})
\tag{3}
\label{eq:image-logprob}
$$
Equation \eqref{eq:image-logprob} uses the pixel probabilities but is still expressed as log-probability for the whole image.
To express these in terms of per-pixel values we need to normalize by the number as pixels $D$
$$
\text{pp} \log \pt(\rvx_{0} | \rvx_{1}^{(i)}) = \log (\prod_{j=1}^D p_j(x_{0j}) | x_{1j}^{(i)})^{1/D} = \frac{1}{D} \sum_{j=1}^D \log p_j(x_{0j} | x_{1j}^{(i)})
$$

### Evaluate prior and denoising matching
We can simplify the evaluation of the *prior* and *denoising matching* terms in \eqref{eq:mc-stable} by writing down the logs of Gaussian pdfs.

The log of univariate Gaussian pdf $f(x) = \gN(x; \mu, \sigma^2)$ is
$$
\log f(x) = \log \left( \frac{1}{\sigma \sqrt{2 \pi}} \exp \left[ -\frac{(x - \mu)^2}{2 \sigma^2} \right] \right)
= -\frac{1}{2}\log(2\pi) - \log(\sigma) - \frac{(x - \mu)^2}{2\sigma^2}.
$$

For two univariate Gaussian pdf $f(x) = \gN(x; \mu_1, \sigma_1^2)$ and $g(x) = \gN(x; \mu_2, \sigma_2^2)$ we have
$$
\log f(x) - \log g(x) 
= -\frac{1}{2}\log(2\pi) - \log(\sigma_1) - \frac{(x - \mu_1)^2}{2\sigma_1^2} + \frac{1}{2}\log(2\pi) + \log(\sigma_2) + \frac{(x - \mu_2)^2}{2\sigma_2^2} 
= \log\frac{\sigma_2}{\sigma_1} + \frac{(x - \mu_2)^2}{2\sigma^2} - \frac{(x - \mu_1)^2}{2\sigma_1^2}
$$
We use this decomposition for evaluating the *prior matching* term with $f = \pt$ and $g = q$.

If furthermore $\sigma_1 = \sigma_2 = \sigma$ we get
$$
\log f(x) - \log g(x) 
= \frac{(x - \mu_2)^2}{2\sigma^2} - \frac{(x - \mu_1)^2}{2\sigma_1^2} 
= \frac{x^2 - 2 x \mu_2 + \mu_2^2 - x^2 + 2 x \mu_1 - \mu_1^2}{2\sigma^2} 
=\frac{x(\mu_1 - \mu_2)}{\sigma^2} + \frac{\mu_2^2 - \mu_1^2}{2\sigma^2} \enspace .
$$
This we can use for the *denoising matching* terms as these have the same variance of $\pt$ and $q$ terms.

### Evaluate data conditional

Given that image pixel values are not continues but discrete $[0, \ldots, 255]$ values, we cannot evaluate the data conditional as a gaussian, but rather discretized gaussian.
I'm not going to explain it here but it is in the implementation and the original inspiration comes from PixelCNN++[^4].

### Negative log-likelihood (NLL)

We are also not typically interested in the likelihood but rather the negative log-likelihood (NLL) and hence get 
$$ \text{pp NLL} = - \text{pp} \log \pt(\rvx_0) \approx - \log \frac{1}{M}\sum_{i=1}^M \exp \frac{1}{D} \left( \log p(\rvx_T^{(i)}) + \sum_{t=1}^T \log \pt(\rvx_{t-1}^{(i)} | \rvx_{t}^{(i)}) - \sum_{t=1}^T \log q(\rvx_t^{(i)} | \rvx_0) \right) \enspace ,
$$
where we can evaluate the log-sum-exp in a stable manner.

### Bits-per-pixel

Finally, we want to move from natural logarithm to log2 to get the restuls in bits

$$BPP = \frac{\text{pp NLL}}{\log 2} = \frac{- \text{pp} \log \pt(\rvx_0)}{\log 2}$$ 

The **number of samples** from the latent should be rather big to get a low-variance estimate of the log-likelihood. For the VAE case in both Importance Weighted VAE [^2] and VampPrior VAE [^3] they used $M = 5000$.


## References

[^1]: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, & H. Lin (Eds.), Advances in Neural Information Processing Systems (Vol. 33, pp. 6840–6851). Curran Associates, Inc. https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf

[^2]: Burda, Y., Grosse, R., & Salakhutdinov, R. (2016). Importance Weighted Autoencoders (arXiv:1509.00519). https://doi.org/10.48550/arXiv.1509.00519

[^3]: Tomczak, J., & Welling, M. (2018). VAE with a VampPrior. Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics, 1214–1223. https://proceedings.mlr.press/v84/tomczak18a.html

[^4]: Salimans, T., Karpathy, A., Chen, X., & Kingma, D. P. (2017). PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications (arXiv:1701.05517). arXiv. https://doi.org/10.48550/arXiv.1701.05517

