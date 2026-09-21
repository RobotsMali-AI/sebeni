"""Optional JAX/Flax policy-update backend."""

from beni.core.srl.jax.policy import JaxPolicyPlugin, jax_uncertainty_scale

__all__ = ["JaxPolicyPlugin", "jax_uncertainty_scale"]
