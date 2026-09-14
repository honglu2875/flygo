"""Embedding derivatives, objective semantics, causal pairs and spherical locality."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import unittest
from dataclasses import replace
import numpy as np
from flygo.fly import FlyConfig,RustFly,PARAMETERS,initialize,firing_rate
from flygo.vision import SphericalRenderer,directions,hex_chart,individual_motor_ports
from flygo.contrastive import BranchObjective
from flygo.data.branches import mine_pairs,probe_families
from tests.test_fly import fixture


class SphericalInputs(unittest.TestCase):
    def test_hex_neighbors_and_explicit_reflection(self):
        qr=np.array([[0,0],[1,0],[0,1],[1,1],[-1,0],[0,-1],[-1,-1]])
        u=hex_chart(qr,origin=[0,0],radians_per_column=.1)
        np.testing.assert_allclose(u[1:]@u[0],np.cos(.1),atol=1e-14)
        mirrored=hex_chart(qr,origin=[0,0],radians_per_column=.1,reflect=True)
        np.testing.assert_array_equal(mirrored,u*np.array([1,-1,1]))
        with self.assertRaisesRegex(ValueError,'hemisphere'):
            hex_chart(qr*20,origin=[0,0],radians_per_column=.1)

    def test_bilateral_history_and_locality(self):
        # Identical columns on the two eyes see different causal history lags.
        a=np.deg2rad([-25]*5)
        u=directions(a,np.deg2rad([-30,30,-30,30,-30]))
        renderer=SphericalRenderer.build(u,np.array(['L','L','R','R','L']))
        x=np.zeros((1,9,9,12),np.float32)
        neutral=renderer.render(x)
        np.testing.assert_array_equal(neutral,np.full((1,5),.5,np.float32))
        for lag,target in [(0,0),(2,1),(1,2),(3,3)]:
            y=x.copy(); y[0,4,4,2*lag]=1
            drive=renderer.render(y)
            self.assertGreater(drive[0,target],.5)
            for other in range(4):
                if other!=target:self.assertEqual(drive[0,other],.5)
            if lag==0:self.assertEqual(drive[0,0],drive[0,4])
        x[:,:,:,:8:2]=1
        full=renderer.render(x,contrast=1.1)
        self.assertTrue(np.all((full>=0)&(full<=1)))
        # Context cannot secretly enter a supposedly visual-only comparison.
        altered=x.copy();altered[...,8:]=.7
        np.testing.assert_array_equal(renderer.render(x),renderer.render(altered))

    def test_uncovered_directions_and_motor_identity(self):
        u=directions([0.],[np.deg2rad(85)])
        r=SphericalRenderer.build(u,np.array(['L']))
        x=np.ones((2,9,9,12),np.float32);x[...,1:8:2]=0
        np.testing.assert_array_equal(r.render(x),np.full((2,1),.5,np.float32))
        ports=individual_motor_ports(9,[1,3],[5,8])
        np.testing.assert_array_equal(ports['output_group'],[-1,-1,-1,-1,-1,0,-1,-1,1])
        with self.assertRaises(ValueError):individual_motor_ports(9,[1,3],[3,8])


class CausalPairs(unittest.TestCase):
    def test_recent_divergence_same_player_and_disjoint_family_split(self):
        prefix=list(range(20));n=30
        games=[]
        for action,quality in [(40,-.8),(41,.7)]:
            games.append(dict(actions=prefix+[action]+list(range(21,30)),raw_value=np.full(n,quality),
                opening_family='shared',split='train',teacher_sha256='teacher'))
        pairs,audit=mine_pairs(games)
        self.assertEqual(audit['pairs'],6)
        for distance,p in enumerate(pairs,1):
            self.assertEqual((p['good'],p['bad'],p['ply'],p['divergence']),(1,0,20+distance,distance))
            self.assertEqual(p['to_play'],1+p['ply']%2)
            self.assertNotIn('split',p)  # The bank freezes a family split after eligibility.
        families=[f'family-{i}' for i in range(12)]
        selected=probe_families(families)
        self.assertEqual(len(selected),3)
        self.assertEqual(selected,probe_families(families[::-1]+families))
        games[1]['actions']=games[0]['actions']
        self.assertFalse(mine_pairs(games)[0])
        games[0]['split']='test'
        with self.assertRaisesRegex(ValueError,'training'):mine_pairs(games)


class EmbeddingLearning(unittest.TestCase):
    def test_objective_against_autodiff_and_quality_direction(self):
        import jax
        import jax.numpy as jnp
        objective=BranchObjective()
        rng=np.random.default_rng(37)
        r=rng.normal(0,.1,(8,7)).astype(np.float32)
        v=rng.uniform(-.5,.5,8).astype(np.float32)
        metrics,dr,dv=objective(r,v)
        loss,grad=jax.value_and_grad(objective.jax_loss,argnums=(0,1))(jnp.array(r),jnp.array(v))
        np.testing.assert_allclose(loss,metrics['loss'],rtol=2e-6)
        np.testing.assert_allclose(grad[0],dr,atol=2e-5,rtol=2e-5)
        np.testing.assert_allclose(grad[1],dv,atol=1e-6,rtol=1e-6)
        self.assertTrue(np.all(dv.reshape(-1,4)[:,:2]<0))
        self.assertTrue(np.all(dv.reshape(-1,4)[:,2:]>0))
        # Complete collapse remains explicitly visible, not a NaN correlation.
        zero,derivative,_=objective(np.zeros((4,7)),np.zeros(4))
        self.assertAlmostEqual(zero['contrastive_loss'],np.log(3))
        self.assertEqual(zero['pair_accuracy'],.5)
        self.assertTrue(np.isfinite(derivative).all())

    def test_native_vjp_all_parameters_updates_and_restore(self):
        import jax
        import jax.numpy as jnp
        from flygo.jax.model import forward
        graph,cfg,ports,params,x,*_=fixture()
        cfg=replace(cfg,rate_softness=.01)
        x=np.concatenate([x,x[::-1]+.05])
        objective=BranchObjective()
        model=RustFly(graph,cfg,ports=ports,params=params)
        metrics,gradient=model.embedding_loss_and_grad(x,objective)
        def loss(p):
            out=forward(p,graph,ports,jnp.array(x),steps=cfg.steps,groups=cfg.groups,actions=cfg.actions,
                        rate_softness=.01,return_embedding=True)
            return objective.jax_loss(out['embedding'],out['score'])
        jl,jg=jax.value_and_grad(loss)({k:jnp.array(v) for k,v in params.items()})
        np.testing.assert_allclose(jl,metrics['loss'],rtol=2e-6)
        for k in PARAMETERS:np.testing.assert_allclose(gradient[k],jg[k],atol=5e-5,rtol=5e-5,err_msg=k)
        before=model.checkpoint_arrays()
        model.train_embedding(x,objective,rate_scales={'readout_gain':0})
        np.testing.assert_array_equal(before['param/readout_gain'],model.parameters()['readout_gain'])
        saved=model.checkpoint_arrays()
        expected=model.train_embedding(x,objective)
        copy=RustFly(graph,cfg,ports=ports,params=params);copy.restore_arrays(saved)
        self.assertEqual(expected,copy.train_embedding(x,objective))
        for k,v in model.checkpoint_arrays().items():np.testing.assert_array_equal(v,copy.checkpoint_arrays()[k])

    def test_stale_and_nonfinite_cotangents_do_not_update(self):
        graph,cfg,ports,params,x,*_=fixture()
        model=RustFly(graph,cfg,ports=ports,params=params)
        data=model._input(x)
        _,_,revision=model.native.embedding(data,cfg.steps)
        saved=model.checkpoint_arrays();model.restore_arrays(saved)
        with self.assertRaisesRegex(ValueError,'Stale'):
            model.native.embedding_backward(data,cfg.steps,np.zeros((cfg.groups,2),np.float32),np.zeros(2,np.float32),revision)
        def invalid(r,v):return {},np.full_like(r,np.nan),np.zeros_like(v)
        with self.assertRaises(ValueError):model.train_embedding(x,invalid)
        for k,v in saved.items():np.testing.assert_array_equal(v,model.checkpoint_arrays()[k])


if __name__=='__main__':unittest.main()
