"""Sparse padding, transpose and edge gradients against independent dense equations."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import unittest
import numpy as np


class SparseJax(unittest.TestCase):
    def test_irregular_graph_forward_and_vjp(self):
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            self.skipTest('JAX optional dependency')
        from flygo.jax.sparse import build_layout,multiply
        rng=np.random.default_rng(517)
        neurons=31
        # Include isolated rows, degree-one rows, bucket boundaries, self edges
        # and shuffled edge IDs. Repeated pairs also have independent gradients.
        destination=np.repeat(np.arange(neurons),np.arange(neurons)%13).astype(np.int32)
        source=rng.integers(neurons,size=len(destination),dtype=np.int32)
        order=rng.permutation(len(source));source=source[order];destination=destination[order]
        weights=rng.normal(size=len(source)).astype(np.float32)
        state=rng.normal(size=(neurons,7)).astype(np.float32)
        cotangent=rng.normal(size=state.shape).astype(np.float32)
        dense=np.zeros((neurons,neurons),np.float32)
        np.add.at(dense,(destination,source),weights)
        for tile in (8,32):
            layout=build_layout(source,destination,neurons,tile_edges=tile)
            actual=jax.jit(multiply)(weights,state,layout)
            np.testing.assert_allclose(actual,dense@state,rtol=2e-5,atol=2e-6)
            fun=lambda w,x:jnp.sum(multiply(w,x,layout)*cotangent)
            gw,gx=jax.jit(jax.grad(fun,argnums=(0,1)))(weights,state)
            np.testing.assert_allclose(gx,dense.T@cotangent,rtol=2e-5,atol=2e-6)
            expected=np.sum(state[source]*cotangent[destination],axis=1)
            np.testing.assert_allclose(gw,expected,rtol=2e-5,atol=2e-6)

    def test_complete_recurrent_gradients(self):
        try:
            import jax
        except ImportError:
            self.skipTest('JAX optional dependency')
        from test_fly import fixture
        from flygo.jax.model import loss
        from flygo.jax.sparse import build_layout
        graph,cfg,ports,params,x,legal,policy,value=fixture()
        kwargs=dict(steps=cfg.steps,groups=cfg.groups,actions=cfg.actions)
        def objective(p,g):return loss(p,g,ports,x,legal.astype(bool),policy,value,**kwargs)[0]
        expected=jax.jit(jax.value_and_grad(objective))(params,graph)
        bucketed={**graph,'layout':build_layout(graph['src'],graph['dst'],len(graph['type_id']),tile_edges=3)}
        actual=jax.jit(jax.value_and_grad(objective))(params,bucketed)
        for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
            np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-7)
