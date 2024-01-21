\(
\def\rv#1{{\mathbf #1}}
\def\pt{p_\theta}
\def\rvx{\rv x}
\def\E{\mathbb{E}}
\def\gN{\mathcal{N}}
\)
# Evaluation of diffusion model

***Created: 2024-01-21***

How can we evaluate the likelihood of a data point $\rvx$ using the diffusion model trained according to (Ho et al., 2020)[^1]?

We start from the importance sampling formulation of our model

$$
\pt(\rvx_0)
= \int q(\rvx_{1:T} | \rvx_0) \frac{\pt(\rvx_{1:T}) \pt(\rvx_0 | \rvx_{1:T})}{q(\rvx_{1:T} | \rvx_0)} \, d\rvx_{1:T}  \enspace .
\tag{1}
\label{eq:importance_sampling}
$$

We normally train it by maximizing a lower bound on the log likelihood $\mathcal{L}(\rvx_0) \le \log \pt(\rvx_0)$
$$
\log \pt(\rvx_0)
= \log \int q(\rvx_{1:T} | \rvx_0) \frac{\pt(\rvx_{0:T})}{q(\rvx_{1:T} | \rvx_0)} \, d\rvx_{1:T}
\ge \int q(\rvx_{1:T} | \rvx_0) \log \frac{\pt(\rvx_{0:T})}{q(\rvx_{1:T} | \rvx_0)} \, d\rvx_{1:T} = \mathcal{L}(\rvx_0) 
 \enspace .
$$

## Monte-carlo estimate
This likelihood \eqref{eq:importance_sampling} can be approximated by a Monte Carlo (MC) estimate as
$$\pt(\rvx_0) \approx \frac{1}{M}\sum_{i=1}^M \frac{\pt(\rvx_0, \rvx_1^{(i)}, \ldots, \rvx_t^{(i)})}{q(\rvx_1^{(i)}, \ldots, \rvx_t^{(i)} | \rvx_0)} \enspace ,
\quad \rvx_1^{(i)}, \ldots, \rvx_t^{(i)} \sim q(\rvx_{1:T} | \rvx_0) \enspace .
\tag{2}
\label{eq:MCestimate}
$$

### Sampling for MC estimate
Sampling out of forward distribution $q(\rvx_{1:T} | \rvx_0)$ is easy.
Thanks to the Markov and Gaussian assumptions it can be shown that 
$$ q(\rvx_{1:T} | \rvx_0) = \prod_{t=1}^T q(\rvx_t | \rvx_{t-1})
= \prod_{t=1}^T q(\rvx_t | \rvx_0)
\enspace ,
\tag{3}
\label{eq:forward}
$$
where the markov conditionals are all Gaussians with known fixed parameters
$q(\rvx_t | \rvx_{t-1}) = \gN(\rvx_t; \sqrt{1-\beta_t}\rvx_{t-1}, \beta_t \rv{I})$
and the $\rvx_0$ are also Gaussians 
$q(\rvx_t | \rvx_0) = \gN(\rvx_t; \sqrt{\bar{a}_t}\rvx_{0}, (1 - \bar{a}_t) \rv{I})$
with $\bar{a}_t = \prod_{s=1}^t (1-\beta_s)$.

> For each $i$ we can sample the $\rvx_1^{(i)}, \ldots, \rvx_t^{(i)}$ independently out of $\rvx_t^{(i)} \sim \gN(\rvx_t; \sqrt{\bar{a}_t}\rvx_0, (1 - \bar{a}_t) \rv{I})$ as $\rvx_t^{(i)} = \sqrt{\bar{a}_t}\rvx_0 + \sqrt{(1 - \bar{a}_t)} \epsilon, \epsilon \sim \gN(\rv{0}, \rv{I})$.

### Evaluation of likelihood terms

For the individual terms in equation \eqref{eq:MCestimate}:
* $q(\rvx_{1:T} | \rvx_0) = \prod_{t=1}^T q(\rvx_t | \rvx_0)$ is simply the product of the Gaussian densities from equation \eqref{eq:forward} which can all be evaluated independently since $\rvx_0$ is fixed.
* $\pt(\rvx_{0:T}) = p(\rvx_{0:T}) = \pt(\rvx_{T}) \prod_{t=1}^T \pt(\rvx_{t-1} | \rvx_{t})$ and we have:
  * $\pt(\rvx_T) = \gN(\rvx_T; \rv{0}, \rv{I})$ is a fixed standard Gaussian
  * $\pt(\rvx_{t-1} | \rvx_{t}) = \gN(\rvx_{t-1}; \frac{1}{\sqrt{\alpha_t}}\rvx_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\rvx_t, t), \sigma^2_t\rv{I})$ with $\sigma_t^2$ being a known (non-learnable) function of the forward variance schedule $\beta_t$ and $\epsilon_\theta(\rvx_t, t)$ is trained to approximate the noise introduced by the forward process.

For a single $i$ for the MC estimate in \eqref{eq:MCestimate} we sample the $\rvx_t^{(i)}$ independently, then evaluate for each the conditionals $q(\rvx_t^{(i)} | \rvx_0)$ and simply take their product to get $q(\rvx_1^{(i)}, \ldots, \rvx_t^{(i)} | \rvx_0)$.
We use the same $\rvx_t^{(i)}$ samples to evaluate each of the reverse conditionals $\pt(\rvx_{t-1}^{(i)} | \rvx_{t}^{(i)})$ (making sure these are the same samples as used for the $q$ evaluation) and get $\pt(\rvx_0, \rvx_1^{(i)}, \ldots, \rvx_t^{(i)})$ again as simple product.

### Log-likelihood stable evaluation and bits-per-dim

Obviously, taking products of distributions is extremely unstable and hence we are better of working in $\log$.
We therefor rewrite the MC estimate as
$$\pt(\rvx_0) \approx
\frac{1}{M}\sum_{i=1}^M \exp \left( \log p(\rvx_T) + \sum_{t=1}^T \log \pt(\rvx_{t-1}^{(i)} | \rvx_{t}^{(i)}) - \sum_{t=1}^T \log q(\rvx_t^{(i)} | \rvx_0) \right) \enspace .
$$

We are also not typically interested in the likelihood but rather the **log-likelihood** and hence get 
$$\log \pt(\rvx_0) \approx
\log \frac{1}{M}\sum_{i=1}^M \exp \left( \log p(\rvx_T) + \sum_{t=1}^T \log \pt(\rvx_{t-1}^{(i)} | \rvx_{t}^{(i)}) - \sum_{t=1}^T \log q(\rvx_t^{(i)} | \rvx_0) \right) \enspace ,
$$
where we can evaluate the log-sum-exp in a stable manner.

**Bits-per-dim** is simply the negative log-likelihood with base 2 normalized by the number of pixels $D$
$$BPP = - \frac{\log_2  \pt(\rvx_0)}{D} = - \frac{\log  \pt(\rvx_0)}{D \log 2}$$ 

The **number of samples** from the latent should be rather big to get a low-variance estimate of the log-likelihood. For the VAE case in both Importance Weighted VAE [^2] and VampPrior VAE [^3] they used $M = 5000$.

There is one more trick that may be necessary: the pixel values are not continuous but discrete 0 to 255 values. Hence assuming that $\pt(\rvx_0 | \rvx_1)$ is Gaussian is strange.
They have dealt with this in PixelCNN++[^4] they have used a mixture of logistic distributions. 
Perhaps this is what the OpenAI implementation does?

## References

[^1]: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, & H. Lin (Eds.), Advances in Neural Information Processing Systems (Vol. 33, pp. 6840–6851). Curran Associates, Inc. https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf

[^2]: Burda, Y., Grosse, R., & Salakhutdinov, R. (2016). Importance Weighted Autoencoders (arXiv:1509.00519). https://doi.org/10.48550/arXiv.1509.00519

[^3]: Tomczak, J., & Welling, M. (2018). VAE with a VampPrior. Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics, 1214–1223. https://proceedings.mlr.press/v84/tomczak18a.html

[^4]: Salimans, T., Karpathy, A., Chen, X., & Kingma, D. P. (2017). PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications (arXiv:1701.05517). arXiv. https://doi.org/10.48550/arXiv.1701.05517

