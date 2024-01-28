\(
\def\rv#1{{\mathbf #1}}
\def\pt{p_\theta}
\def\rvx{\rv x}
\def\E{\mathbb{E}}
\def\gN{\mathcal{N}}
\)
# Evaluation of diffusion model

***Created: 2024-01-21***
***Updated: 2024-01-26***


How can we evaluate the likelihood of a data point $\rvx$ using the diffusion model trained according to (Ho et al., 2020)[^1]?

We start from the importance sampling formulation of our model

$$
\pt(\rvx_0)
= \int q(\rvx_{1:T} | \rvx_0) \frac{\pt(\rvx_{1:T}) \pt(\rvx_0 | \rvx_{1:T})}{q(\rvx_{1:T} | \rvx_0)} \, d\rvx_{1:T}  \enspace .
\tag{1}
\label{eq:importance_sampling}
$$

<!-- 
We normally train it by maximizing a lower bound on the log likelihood $\mathcal{L}(\rvx_0) \le \log \pt(\rvx_0)$
$$
\log \pt(\rvx_0)
= \log \int q(\rvx_{1:T} | \rvx_0) \frac{\pt(\rvx_{0:T})}{q(\rvx_{1:T} | \rvx_0)} \, d\rvx_{1:T}
\ge \int q(\rvx_{1:T} | \rvx_0) \log \frac{\pt(\rvx_{0:T})}{q(\rvx_{1:T} | \rvx_0)} \, d\rvx_{1:T} = \mathcal{L}(\rvx_0) 
 \enspace .
$$
 -->
## Monte-carlo estimate
This likelihood \eqref{eq:importance_sampling} can be approximated by a Monte Carlo (MC) estimate as
$$\pt(\rvx_0) \approx \frac{1}{M}\sum_{i=1}^M \frac{\pt(\rvx_0, \rvx_1^{(i)}, \ldots, \rvx_T^{(i)})}{q(\rvx_1^{(i)}, \ldots, \rvx_T^{(i)} | \rvx_0)} \enspace ,
\quad \rvx_1^{(i)}, \ldots, \rvx_T^{(i)} \sim q(\rvx_{1:T} | \rvx_0) \enspace .
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
* alternatively we can use $q(\rvx_{1:T} | \rvx_0) = \prod_{t=1}^T q(\rvx_t | \rvx_{t-1})$
* $\pt(\rvx_{0:T}) = p(\rvx_{0:T}) = \pt(\rvx_{T}) \prod_{t=1}^T \pt(\rvx_{t-1} | \rvx_{t})$ and we have:
  * $\pt(\rvx_T) = \gN(\rvx_T; \rv{0}, \rv{I})$ is a fixed standard Gaussian
  * $\pt(\rvx_{t-1} | \rvx_{t}) = \gN(\rvx_{t-1}; \frac{1}{\sqrt{\alpha_t}}\left(\rvx_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\rvx_t, t)\right), \sigma^2_t\rv{I})$ with $\sigma_t^2$ being a known (non-learnable) function of the forward variance schedule $\beta_t$ and $\epsilon_\theta(\rvx_t, t)$ is trained to approximate the noise introduced by the forward process. Here we will use $\sigma_t^2 = \frac{\beta_t(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$ 

So that
$$\pt(\rvx_0) \approx \frac{1}{M}\sum_{i=1}^M \frac{\pt(\rvx_T) \pt(\rvx_{0} | \rvx_{1}) \prod_{t=2}^T \pt(\rvx_{t-1} | \rvx_{t})}{\prod_{t=1}^T q(\rvx_t | \rvx_{t-1})} \enspace ,
\quad \rvx_t^{(i)} \sim q(\rvx_{t} | \rvx_0) \ \forall t=1, \ldots, T \enspace .
\tag{4}
\label{eq:MCestimate_details}
$$

For a single $i$ for the MC estimate in \eqref{eq:MCestimate_details} we sample the $\rvx_t^{(i)}$ independently, then evaluate for each the conditionals $q(\rvx_t^{(i)} | \rvx_0)$ or $q(\rvx_t | \rvx_{t-1})$ and simply take their product to get $q(\rvx_1^{(i)}, \ldots, \rvx_t^{(i)} | \rvx_0)$.
We use the same $\rvx_t^{(i)}$ samples to evaluate each of the reverse conditionals $\pt(\rvx_{t-1}^{(i)} | \rvx_{t}^{(i)})$ (making sure these are the same samples as used for the $q$ evaluation) and get $\pt(\rvx_0, \rvx_1^{(i)}, \ldots, \rvx_t^{(i)})$ again as simple product.

Yet another option would be to switch to the same ordering as in the reverse process.
Starting from 
$$
q(\rvx_t | \rvx_{t-1}) = q(\rvx_t | \rvx_{t-1}, \rvx_0) = \frac{q(\rvx_{t-1} | \rvx_{t}, \rvx_0) q(\rvx_{t} | \rvx_0) q(\rvx_0)}{q(\rvx_{t-1} | \rvx_0) q(\rvx_0)}
$$
and we get
$$
\prod_{t=1}^T q(\rvx_t | \rvx_{t-1}) = q(\rvx_1 | \rvx_0) \prod_{t=2}^T \frac{q(\rvx_{t-1} | \rvx_{t}, \rvx_0) q(\rvx_{t} | \rvx_0) q(\rvx_0)}{q(\rvx_{t-1} | \rvx_0) q(\rvx_0)}
= q(\rvx_1 | \rvx_0) \frac{q(\rvx_{T} | \rvx_0)}{q(\rvx_1 | \rvx_0)} \prod_{t=2}^T q(\rvx_{t-1} | \rvx_{t}, \rvx_0)
= q(\rvx_{T} | \rvx_0) \prod_{t=2}^T q(\rvx_{t-1} | \rvx_{t}, \rvx_0)
$$

Bringing this back to importance sampling
$$\pt(\rvx_0) \approx \frac{1}{M}\sum_{i=1}^M \frac{\pt(\rvx_T) \pt(\rvx_{0} | \rvx_{1}) \prod_{t=2}^T \pt(\rvx_{t-1} | \rvx_{t})}{q(\rvx_{T} | \rvx_0) \prod_{t=2}^T q(\rvx_{t-1} | \rvx_{t}, \rvx_0)}  
= \frac{1}{M}\sum_{i=1}^M \pt(\rvx_{0} | \rvx_{1}) \frac{\pt(\rvx_T)}{q(\rvx_{T} | \rvx_0)}
\prod_{t=2}^T \frac{\pt(\rvx_{t-1} | \rvx_{t})}{q(\rvx_{t-1} | \rvx_{t}, \rvx_0)}
\enspace ,
\tag{5}
\label{eq:MCestimate_details2}
$$
where 
$q(\rvx_{t-1} | \rvx_{t}, \rvx_0) = \gN(\rvx_{t-1}; \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\rvx_t + \frac{(1-\alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{(1 - \bar{\alpha}_t)} \rvx_0, \frac{\beta_t(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \rv{I})$

### Log-likelihood stable evaluation

Obviously, taking products of distributions is extremely unstable and hence we are better of working in $\log$.
We therefore rewrite the MC estimate from equation \eqref{eq:MCestimate_details} using logs and use $\prod_{t=1}^T q(\rvx_t | \rvx_0)$ in the denominator (this could be readily replace by $\prod_{t=1}^T q(\rvx_t | \rvx_{t-1})$).
$$\pt(\rvx_0) \approx
\frac{1}{M}\sum_{i=1}^M \exp \left( \log p(\rvx_T^{(i)}) + \sum_{t=1}^T \log \pt(\rvx_{t-1}^{(i)} | \rvx_{t}^{(i)}) - \sum_{t=1}^T \log q(\rvx_t^{(i)} | \rvx_0) \right) \enspace .
$$

Instead of using the above it may be more convenient to start from \eqref{eq:MCestimate_details2} and by introducing logs we get 
$$\pt(\rvx_0) \approx
\frac{1}{M}\sum_{i=1}^M \exp \left( \underbrace{\log \pt(\rvx_{0} | \rvx_{1}^{(i)})}_{P0} + \underbrace{\log \pt(\rvx_T^{(i)}) - \log q(\rvx_{T}^{(i)} | \rvx_0)}_{PT} + \sum_{t=2}^T \left[ \underbrace{\log \pt(\rvx_{t-1}^{(i)} | \rvx_{t}^{(i)}) - \log q(\rvx_{t-1}^{(i)} | \rvx_{t}^{(i)}, \rvx_0)}_{Pt} \right] \right)
\enspace .
\tag{6}
\label{eq:loglik}
$$

For a D-variate gaussian pdf $f(\rvx) = \gN(\rvx; \rv{\mu}, \sigma^2 \rv{I})$ we have
$$
\log f(\rvx) = \log \left( (2 \pi)^{-D/2} \sigma^{-D}  \exp \left[ -||\rvx - \rv{\mu}||_2^2/(2 \sigma^2) \right] \right)
= -\frac{D}{2}\log(2\pi) -D\log(\sigma) - \frac{||\rvx - \rv{\mu}||_2^2}{2\sigma^2}.
$$

For two D-variate gaussian pdf $f(\rvx) = \gN(\rvx; \rv{\mu}_1, \sigma^2_1 \rv{I})$ and $g(\rvx) = \gN(\rvx; \rv{\mu}_2, \sigma^2_2 \rv{I})$ we have

$$
\begin{align}
\log f(\rvx) - \log g(\rvx) 
=& -\frac{D}{2}\log(2\pi) -D\log(\sigma_1) - \frac{||\rvx - \rv{\mu}_1||_2^2}{2\sigma_1^2} + \frac{D}{2}\log(2\pi) +D\log(\sigma_2) + \frac{||\rvx - \rv{\mu}_2||_2^2}{2\sigma_2^2} \\
=& D \log\frac{\sigma_2}{\sigma_1} + \frac{||\rvx - \rv{\mu}_2||_2^2}{2\sigma_2^2} - \frac{||\rvx - \rv{\mu}_1||_2^2}{2\sigma_1^2} \\
\end{align}
$$

If furthermore $\sigma_1 = \sigma_2 = \sigma$ we get
$$
\begin{align}
\log f(\rvx) - \log g(\rvx) 
=& \frac{||\rvx - \rv{\mu}_2||_2^2}{2\sigma^2} - \frac{||\rvx - \rv{\mu}_1||_2^2}{2\sigma^2} \\
=& \frac{\rvx^T \rvx - 2 \rvx^T \mu_2 + \mu_2^T \mu_2 - \rvx^T \rvx + 2 \rvx^T \mu_1 - \mu_1^T \mu_1}{2\sigma^2} \\
=& \rvx^T(\mu_1 - \mu_2) + \frac{||\mu_2||_2^2 - ||\mu_1||_2^2}{2\sigma^2} \\
\end{align}
$$



<!-- For the terms in \eqref{eq:loglik} we get
$$
PT = -\frac{D}{2}\log(2\pi) -D\log(1) - \frac{||\rvx_T - \rv{0}||_2^2}{2 \cdot 1} + \frac{D}{2}\log(2\pi) + D \log(\sqrt{(1 - \bar{a}_T)}) + \frac{||\rvx_T - \sqrt{\bar{a}_t}\rvx_{0}||_2^2}{2(1 - \bar{a}_t)} 
= D \log(\sqrt{(1 - \bar{a}_T)}) + \frac{||\rvx_T - \sqrt{\bar{a}_t}\rvx_{0}||_2^2}{2(1 - \bar{a}_t)}  - \frac{||\rvx_T||_2^2}{2}
$$
<!-- $$
= D \log(\sqrt{(1 - \bar{a}_T)}) + \frac{\rvx_T^T\rvx_T - 2\rvx_T^T\sqrt{\bar{a}_t}\rvx_{0} + \bar{a}_t\rvx_0^T\rvx_0 - (1 - \bar{a}_t)\rvx_T^T\rvx_T}{2(1 - \bar{a}_t)}
= D \log(\sqrt{(1 - \bar{a}_T)}) + \frac{\bar{a}_t (\rvx_T^T\rvx_T + \rvx_0^T\rvx_0) - 2\sqrt{\bar{a}_t}\rvx_T^T\rvx_{0}}{2(1 - \bar{a}_t)}
$$ -->
<!--
Further note that $\pt(\rvx_{t-1} | \rvx_{t})$ and $q(\rvx_{t-1} | \rvx_{t}, \rvx_0)$ have the same variance which we indicate as $\sigma_t^2 = \frac{\beta_t(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$ and thus we have
$$
\begin{align}
Pt =& -\frac{D}{2}\log(2\pi) - D\log(\sigma_t) - \frac{||\rvx_{t-1} - \frac{1}{\sqrt{\alpha_t}}\left(\rvx_{t} - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\rvx_t, t)\right)||_2^2}{2 \sigma_t^2} + 
\frac{D}{2}\log(2\pi) + D\log(\sigma_t) + \frac{||\rvx_{t-1} - (\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\rvx_t + \frac{(1-\alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{(1 - \bar{\alpha}_t)} \rvx_0)||_2^2}{2 \sigma_t^2} \\
=& \frac{||\rvx_{t-1} - (\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\rvx_t + \frac{(1-\alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{(1 - \bar{\alpha}_t)} \rvx_0)||_2^2 - ||\rvx_{t-1} - \frac{1}{\sqrt{\alpha_t}}\left(\rvx_{t} - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\rvx_t, t)\right)||_2^2}{2 \sigma_t^2} \\
\end{align}
$$ -->
<!-- $$
\begin{align}
=& \frac{-2\rvx_{t-1}^T(\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\rvx_t + \frac{(1-\alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{(1 - \bar{\alpha}_t)} \rvx_0) + 
||\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\rvx_t + \frac{(1-\alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{(1 - \bar{\alpha}_t)} \rvx_0||_2^2 +  
\frac{2}{\sqrt{\alpha_t}}\rvx_{t-1}^T\left(\rvx_{t} - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\rvx_t, t)\right) - 
\frac{1}{\alpha_t} ||\rvx_{t} - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\rvx_t, t) ||_2^2}{2 \sigma_t^2} \\
=& \frac{(\frac{2}{\sqrt{\alpha_t}}-\frac{2\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t})\rvx_{t-1}^T\rvx_t -
\frac{2(1-\alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{(1 - \bar{\alpha}_t)} \rvx_{t-1}^T\rvx_0 +
\frac{\alpha_t(1 - \bar{\alpha}_{t-1})^2}{(1 - \bar{\alpha}_t)^2}\rvx_t^T\rvx_t +
\frac{\sqrt{\alpha_t \bar{\alpha}_{t-1}}(1 - \bar{\alpha}_{t-1})(1-\alpha_t)}{(1 - \bar{\alpha}_t)^2}\rvx_t\rvx_0 -
\frac{2(1-\alpha_t)}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\rvx_{t-1}^T\epsilon_\theta(\rvx_t, t) -
\frac{1}{\alpha_t} \rvx_t^T \rvx_t + 
\frac{1-\alpha_t}{\alpha_t \sqrt{1-\bar\alpha_t}}\rvx_t^T \epsilon_\theta(\rvx_t, t) - 
\frac{(1-\alpha_t)^2}{1-\bar\alpha_t}\epsilon_\theta(\rvx_t, t)^T \epsilon_\theta(\rvx_t, t)}
{2 \sigma_t^2} \\
=& \frac{\frac{2(1 - \bar{\alpha}_t) - 2\alpha_t(1 - \bar{\alpha}_{t-1})}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)}\rvx_{t-1}^T\rvx_t -
\frac{2(1-\alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{(1 - \bar{\alpha}_t)} \rvx_{t-1}^T\rvx_0 +
\frac{\alpha_t^2(1 - \bar{\alpha}_{t-1})^2 - (1 - \bar{\alpha}_t)^2}{\alpha_t(1 - \bar{\alpha}_t)^2}\rvx_t^T\rvx_t +
\frac{\sqrt{\alpha_t \bar{\alpha}_{t-1}}(1 - \bar{\alpha}_{t-1})(1-\alpha_t)}{(1 - \bar{\alpha}_t)^2}\rvx_t\rvx_0 -
\frac{2(1-\alpha_t)}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\rvx_{t-1}^T\epsilon_\theta(\rvx_t, t) -
\frac{1-\alpha_t}{\alpha_t \sqrt{1-\bar\alpha_t}}\rvx_t^T \epsilon_\theta(\rvx_t, t) - 
\frac{(1-\alpha_t)^2}{1-\bar\alpha_t}\epsilon_\theta(\rvx_t, t)^T \epsilon_\theta(\rvx_t, t)}
{2 \sigma_t^2} \\
= & \frac{\frac{2(1-\alpha_t)}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)}\rvx_{t-1}^T\rvx_t - 
\frac{2(1-\alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{(1 - \bar{\alpha}_t)} \rvx_{t-1}^T\rvx_0 + 
\frac{\alpha_t^2(1 - \bar{\alpha}_{t-1})^2 - (1 - \bar{\alpha}_t)^2}{\alpha_t(1 - \bar{\alpha}_t)^2}\rvx_t^T\rvx_t +
\frac{\sqrt{\alpha_t \bar{\alpha}_{t-1}}(1 - \bar{\alpha}_{t-1})(1-\alpha_t)}{(1 - \bar{\alpha}_t)^2}\rvx_t\rvx_0 -
\frac{2(1-\alpha_t)\sqrt{1-\bar\alpha_t}}{\sqrt{\alpha_t}(1-\bar\alpha_t)}\rvx_{t-1}^T\epsilon_\theta(\rvx_t, t) -
\frac{(1-\alpha_t)\sqrt{1-\bar\alpha_t}}{\alpha_t (1-\bar\alpha_t)}\rvx_t^T \epsilon_\theta(\rvx_t, t) - 
\frac{(1-\alpha_t)^2}{1-\bar\alpha_t}\epsilon_\theta(\rvx_t, t)^T \epsilon_\theta(\rvx_t, t)}
{2 \frac{(1-\alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}} \\
= & \frac{1}{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}\rvx_{t-1}^T\rvx_t -
\frac{\sqrt{\bar{\alpha}_{t-1}}}{(1 - \bar{\alpha}_{t-1})} \rvx_{t-1}^T\rvx_0 + 
\frac{\alpha_t^2(1 - \bar{\alpha}_{t-1})^2 - (1 - \bar{\alpha}_t)^2}{2(1 - \bar{\alpha}_t)^2(1-\alpha_t)}\rvx_t^T\rvx_t +
\frac{\sqrt{\alpha_t \bar{\alpha}_{t-1}}}{2(1 - \bar{\alpha}_t)}\rvx_t\rvx_0 -
\frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}\rvx_{t-1}^T\epsilon_\theta(\rvx_t, t) -
\frac{\sqrt{1-\bar\alpha_t}}{2 \alpha_t (1 - \bar{\alpha}_{t-1})}\rvx_t^T \epsilon_\theta(\rvx_t, t) - 
\frac{(1-\alpha_t)}{2(1 - \bar{\alpha}_{t-1})}\epsilon_\theta(\rvx_t, t)^T \epsilon_\theta(\rvx_t, t)
\end{align}
$$ -->

For implementation, remember that $\rv{a}^T \rv{b} = \sum_i^d a_i b_i$ so these are all trivial element-wise products.

### Bits-per-dim

We are also not typically interested in the likelihood but rather the **log-likelihood** and hence get 
$$\log \pt(\rvx_0) \approx
\log \frac{1}{M}\sum_{i=1}^M \exp \left( \log p(\rvx_T^{(i)}) + \sum_{t=1}^T \log \pt(\rvx_{t-1}^{(i)} | \rvx_{t}^{(i)}) - \sum_{t=1}^T \log q(\rvx_t^{(i)} | \rvx_0) \right) \enspace ,
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

