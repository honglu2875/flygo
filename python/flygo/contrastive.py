"""Quality-aware, within-branch contrastive objective; no action groups are fitted.

Each consecutive quartet is [better view A, better view B, worse A, worse B].
Views change global image contrast, not stones, orientation or history.
Only the two views of the paired branch are negatives: unrelated games are not.
"""
from dataclasses import dataclass
import numpy as np


VERSION = 'branch-contrast-and-linear-rank-v2'


@dataclass(frozen=True)
class BranchObjective:
    temperature: float = .1
    rank_temperature: float = .25
    rank_weight: float = 1.
    norm_floor: float = 1e-3

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (
                self.temperature,self.rank_temperature,self.rank_weight,self.norm_floor)):
            raise ValueError('Objective settings must be finite and positive')

    def __call__(self, embedding, value):
        """FP64 small loss, FP32 cotangents; value is an unbounded ordinal score."""
        r = np.asarray(embedding,np.float64); v = np.asarray(value,np.float64)
        if (r.ndim != 2 or not len(r) or len(r)%4 or v.shape != (len(r),)
                or not np.isfinite(r).all() or not np.isfinite(v).all()):
            raise ValueError('Expected finite consecutive better/better/worse/worse quartets')
        batch, groups = r.shape
        # Smooth norm floor bounds derivatives without dropping or whitening cells.
        norm = np.sqrt(np.square(r).sum(axis=1,keepdims=True) + self.norm_floor**2)
        z = (r/norm).reshape(-1,4,groups)
        similarity = z @ z.transpose(0,2,1) / self.temperature
        similarity[:,np.arange(4),np.arange(4)] = -np.inf
        maximum = similarity.max(axis=2,keepdims=True)
        exp = np.exp(similarity-maximum)
        prob = exp/exp.sum(axis=2,keepdims=True)
        pos = np.array([1,0,3,2])
        logp = similarity-maximum-np.log(exp.sum(axis=2,keepdims=True))
        contrastive = -logp[:,np.arange(4),pos].mean()
        dsim = prob.copy(); dsim[:,np.arange(4),pos] -= 1
        dsim /= batch*self.temperature
        dz = ((dsim+dsim.transpose(0,2,1)) @ z).reshape(batch,groups)
        unit = z.reshape(batch,groups)
        dr = (dz-unit*(dz*unit).sum(axis=1,keepdims=True))/norm
        scores = v.reshape(-1,4)
        delta = scores[:,:2].mean(axis=1)-scores[:,2:].mean(axis=1)
        t = -delta/self.rank_temperature
        ranking = np.logaddexp(0.,t).mean()
        slope = -np.exp(-np.logaddexp(0.,-t)) / (len(scores)*self.rank_temperature)
        dv = np.empty_like(scores)
        dv[:,:2] = slope[:,None]/2; dv[:,2:] = -slope[:,None]/2
        metrics = dict(loss=float(contrastive+self.rank_weight*ranking),
            contrastive_loss=float(contrastive), ranking_loss=float(ranking),
            pair_accuracy=float(np.mean((delta>0)+.5*(delta==0))),
            pair_tie_fraction=float(np.mean(delta==0)),mean_score_gap=float(delta.mean()),
            rms_rate=float(np.sqrt(np.square(r).mean())))
        return metrics, dr.astype(np.float32), (self.rank_weight*dv).astype(np.float32).ravel()

    def jax_loss(self, embedding, value):
        """Independent autodiff reference; use on CPU before a new TPU path."""
        import jax.numpy as jnp
        import jax
        norm = jnp.sqrt(jnp.square(embedding).sum(axis=1,keepdims=True)+self.norm_floor**2)
        z = (embedding/norm).reshape(-1,4,embedding.shape[1])
        sim = jnp.einsum('qig,qjg->qij',z,z,precision=jax.lax.Precision.HIGHEST)/self.temperature
        sim = jnp.where(jnp.eye(4,dtype=bool)[None],-jnp.inf,sim)
        logp = jax.nn.log_softmax(sim,axis=-1)
        contrast = -logp[:,jnp.arange(4),jnp.array([1,0,3,2])].mean()
        score = value.reshape(-1,4)
        delta = score[:,:2].mean(axis=1)-score[:,2:].mean(axis=1)
        return contrast+self.rank_weight*jax.nn.softplus(-delta/self.rank_temperature).mean()
