import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from herculens import MassModel

jax.config.update("jax_enable_x64", True)


class MassModelMassSheet(MassModel):

    def __init__(self, profile_list, use_jax_scan=False, verbose=False, **kwargs):
        super().__init__(profile_list, use_jax_scan=use_jax_scan, verbose=verbose)
        # Add mass sheet convergence as an attribute
        self.kappa0 = kwargs.get('kappa0', 0.0)  # default value for mass sheet convergence

    def potential(self, x, y, kwargs, k=None):
        psi = super().potential(x, y, kwargs, k=k)
        return (1 - self.kappa0) * psi + 0.5 * self.kappa0 * (x**2 + y**2)

    def alpha(self, x, y, kwargs, k=None):
        ax, ay = super().alpha(x, y, kwargs, k=k)
        return (1 - self.kappa0) * ax + self.kappa0 * x, (1 - self.kappa0) * ay + self.kappa0 * y

    def hessian(self, x, y, kwargs, k=None):
        f_xx, f_xy, f_yx, f_yy = super().hessian(x, y, kwargs, k=k)
        return (1 - self.kappa0) * f_xx + self.kappa0, (1 - self.kappa0) * f_xy, (1 - self.kappa0) * f_yx, (1 - self.kappa0) * f_yy + self.kappa0

    def kappa(self, x, y, kwargs, k=None):
        return (1 - self.kappa0) * super().kappa(x, y, kwargs, k=k) + self.kappa0

