import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from functools import partial

from typing import Optional, Sequence, Union, Callable
from jaxtyping import PyTree, Array


# Score matching objectives


# Flow matching objectives


@partial(
    jax.jit,
    static_argnames=(
        "model_fn", "mean_fn", "std_fn", "weight_fn",
        "rebalance_loss", "axis", "score_parameterization", "estimate_score",
    ),
)
def conditional_flow_and_score_matching_loss(
    params: PyTree,
    key: PRNGKey,
    times: Array,
    xs_source: Array,
    xs_target: Array,
    model_fn: Callable,
    mean_fn: Callable,
    std_fn: Callable,
    *args,
    loss_mask: Optional[Array] = None,
    weight_fn: Optional[Callable] = None,
    rebalance_loss: bool = False,
    axis: int = -1,
    score_parameterization: str = "raw",
    estimate_score: bool = False,
    eps: Optional[Array] = None,
    **kwargs,
):
    """Conditional flow matching loss, optionally with a co-trained score head.

    By setting estimate_score to False, only the conditional flow matching
    (velocity) loss is computed; with estimate_score=True both the velocity loss
    and a score/denoiser loss are computed (the stochastic-interpolant dual head).

    The five additions below are no-op-defaulted: with `loss_mask=None,
    weight_fn=None, rebalance_loss=False, axis=-1, score_parameterization="raw"`
    the result is byte-identical to the original (velocity loss
    `mean(sum((v-u)^2, -1))`; raw score loss `mean(sum((s + eps/std)^2, -1))`).
    They mirror `denoising_score_matching_loss` so simreturn's objectives seam can
    drive FM-velocity (A1) and dual-head (A2) training through this one function.

    Args:
        params (PyTree): Parameters of the model_fn given as a PyTree.
        key (PRNGKey): Random key.
        times (Array): Time points, should be broadcastable to shape (batch_size, 1).
        xs_source (Array): Marginal distribution at time t=0, refered to as source distribution.
        xs_target (Array): Marginal distribution at time t, refered to as target distribution.
        model_fn (Callable): Model_fn that takes parameters, times, and samples as input and returns the vector field and optionally the marginal score. Should be a function of the form model_fn(params, times, xs_t, *args, **kwargs) -> v_t(, s_t).
        mean_fn (Callable): The mean function of the Gaussian probability path (3-arg: mean_fn(xs_source, xs_target, t)).
        std_fn (Callable): The standard deviation function of the Gaussian probability path (3-arg: std_fn(xs_source, xs_target, t)).
        loss_mask (Optional[Array]): Conditioned-token bridge mask, broadcastable to xs_target. Where True, the token is pinned to xs_target at every t (fed clean to the model) and its residual is zeroed in BOTH the velocity and score terms. Mirrors the all-conditional DSM mask.
        weight_fn (Optional[Callable]): Per-sample weight `weight_fn(times)` multiplying the summed residual. None => unit weight.
        rebalance_loss (bool): If True, divide the per-sample summed loss by the count of non-masked tokens along `axis` (occupancy normalisation). Requires loss_mask.
        axis (int): Axis to sum the residual over. simreturn passes axis=-2 (the token axis); the standalone default is -1.
        score_parameterization (str): "raw" keeps the divergent score target -eps/std (the score head learns the marginal score directly). "denoiser" switches the score-head target to the BOUNDED eps-prediction (the head learns eta ~= E[eps|x_t]); the score is recovered analytically as s = -eta/std at inference. Only consulted when estimate_score=True.
        estimate_score (bool, optional): If set to true, both flow and score matching objectives are computed. Defaults to False.
        eps (Optional[Array]): The path Gaussian noise. None (default) draws it from `key` (the original behaviour). When provided (e.g. a minibatch-OT-reordered source for Multisample flow matching) it is used verbatim as the bridge noise, so the (source, target) coupling carries through to both the velocity target and the score/denoiser target. Must broadcast to xs_source.shape.

    Returns:
        (loss_flow, Optional[loss_score]): Respective loss functions
    """
    # Sample x_t along the conditional Gaussian path. `eps` defaults to a fresh
    # Gaussian draw from `key`; an externally supplied `eps` (the coupled source)
    # overrides it without changing any other behaviour.
    if eps is None:
        eps = jax.random.normal(key, shape=xs_source.shape)
    xs_t = (
        mean_fn(xs_source, xs_target, times) + std_fn(xs_source, xs_target, times) * eps
    )

    # Conditioned-token degenerate bridge: pin masked tokens to their target
    # value at every t (they are fed clean to the model so the prediction on
    # latent tokens is conditional on them). Their residual is zeroed below.
    if loss_mask is not None:
        loss_mask = loss_mask.reshape(xs_target.shape)
        xs_t = jnp.where(loss_mask, xs_target, xs_t)

    # Compute u_t -> the conditional velocity target (valid for Gaussian paths).
    t = jnp.broadcast_to(
        times, xs_target.shape
    )  # Pad to x shape for jax.grad -> x.shape
    std_fn_grad = jax.grad(lambda x_s, x_t, t: std_fn(x_s, x_t, t).sum(), argnums=2)
    mean_fn_grad = jax.grad(lambda x_s, x_t, t: mean_fn(x_s, x_t, t).sum(), argnums=2)
    u_t = std_fn_grad(xs_source, xs_target, t) * eps + mean_fn_grad(
        xs_source, xs_target, t
    )

    def _reduce(residual):
        """Mask -> sum over `axis` -> weight -> rebalance -> mean. No-op for defaults."""
        if loss_mask is not None:
            residual = jnp.where(loss_mask, 0.0, residual)
        out = jnp.sum(residual, axis=axis, keepdims=True)
        if weight_fn is not None:
            out = weight_fn(times) * out
        if rebalance_loss and loss_mask is not None:
            num_elements = jnp.sum(~loss_mask, axis=axis, keepdims=True)
            out = jnp.where(num_elements > 0, out / num_elements, 0.0)
        return jnp.mean(out)

    if not estimate_score:
        # Velocity-only (FM) -> the network predicts a velocity.
        v_t = model_fn(params, times, xs_t, *args, **kwargs)
        return _reduce((v_t - u_t) ** 2)

    # Dual head -> the network predicts (velocity, score-or-denoiser).
    v_t, s_t = model_fn(params, times, xs_t, *args, **kwargs)
    loss = _reduce((v_t - u_t) ** 2)
    if score_parameterization == "denoiser":
        # Bounded eps-prediction target (the endpoint fix); s_t is eta ~= E[eps|x_t].
        score_target = eps
    else:
        # Raw score target -eps/std (diverges as std -> 0); preserves legacy behaviour.
        score_target = -eps / std_fn(xs_source, xs_target, times)
    loss_score = _reduce((s_t - score_target) ** 2)
    return loss, loss_score


def denoising_score_matching_loss(
    params: PyTree,
    key: PRNGKey,
    times: Array,
    xs_target: Array,
    loss_mask: Optional[Array],
    *args,
    model_fn: Callable,
    mean_fn: Callable,
    std_fn: Callable,
    weight_fn: Callable,
    axis: int = -2,
    rebalance_loss: bool = False,
    **kwargs,
) -> Array:
    """This function computes the denoising score matching loss. Which can be used to train diffusion models.

    Args:
        params (PyTree): Parameters of the model_fn given as a PyTree.
        key (PRNGKey): Random generator key.
        times (Array): Time points, should be broadcastable to shape (batch_size, 1).
        xs_target (Array): Target distribution.
        loss_mask (Optional[Array]): Mask for the target distribution. If None, no mask is applied, should be broadcastable to shape (batch_size, 1).
        model_fn (Callable): Score model that takes parameters, times, and samples as input and returns the score. Should be a function of the form model_fn(params, times, xs_t, *args) -> s_t.
        mean_fn (Callable): Mean function of the SDE.
        std_fn (Callable): Std function of the SDE.
        weight_fn (Callable): Weight function for the loss.
        axis (int, optional): Axis to sum over. Defaults to -2.
        

    Returns:
        Array: Loss
    """
    eps = jax.random.normal(key, shape=xs_target.shape)
    mean_t = mean_fn(times, xs_target)
    std_t = std_fn(times, xs_target)
    xs_t = mean_t + std_t * eps
    
    if loss_mask is not None:
        loss_mask = loss_mask.reshape(xs_target.shape)
        xs_t = jnp.where(loss_mask, xs_target, xs_t)
    
    score_pred = model_fn(params, times, xs_t, *args, **kwargs)
    score_target = -eps / std_t

    loss = (score_pred - score_target) ** 2
    if loss_mask is not None:
        loss = jnp.where(loss_mask, 0.0,loss)
    loss = weight_fn(times) * jnp.sum(loss, axis=axis, keepdims=True)
    if rebalance_loss:
        num_elements = jnp.sum(~loss_mask, axis=axis, keepdims=True)
        loss = jnp.where(num_elements > 0, loss / num_elements, 0.0)
    loss = jnp.mean(loss)

    return loss


def score_matching_loss(
    params: PyTree,
    times: Array,
    xs_target: Array,
    mask: Optional[Array],
    *args,
    model_fn: Callable,
):
    """Score matching loss. Minimizing the Fisher divergence between the model and the target distribution, using partial integration trick.

    NOTE: This becomes inefficient when the dimension of the target distribution is high, as the Jacobian of the model is computed.

    Args:
        params (PyTree): Parameters of the model_fn given as a PyTree.
        times (Array): Time points, should be broadcastable to shape (batch_size, 1).
        xs_target (Array): Target distribution.
        mask (Optional[Array]): _description_
        model_fn (Callable): _description_

    Returns:
        _type_: _description_
    """
    jac_model_fn = jax.jacfwd(model_fn, argnums=2)
    score = model_fn(params, times, xs_target, *args)
    jac_score = jac_model_fn(params, times, xs_target, *args)
    loss = jnp.trace(jac_score) + jnp.sum(score**2, axis=-1)
    loss = jnp.mean(jnp.where(mask, loss, 0.0))
    return loss


def sliced_score_matching(
    params: PyTree,
    key: PRNGKey,
    times: Array,
    xs_target: Array,
    mask: Optional[Array],
    *args,
    model_fn: Callable,
    num_slices: int = 1,
):
    def _f(x, v):
        val, grad = jax.value_and_grad(
            lambda x, v: jnp.sum(model_fn(params, times, xs_target, *args) * v)
        )(x, v)
        grad = jnp.sum(grad * v)
        return val, grad

    _f = jax.vmap(jax.vmap(_f, in_axes=(None, 0)), in_axes=(0, None))

    # Slice directions
    v = jax.random.normal(key, shape=(num_slices, xs_target.shape[-1]))
    sliced_score, jac_trace = _f(xs_target, v)
    loss = jnp.sum(sliced_score**2, -1) + jnp.mean(jac_trace, -1)
    loss = jnp.mean(jnp.where(mask, loss, 0.0))
    return loss
