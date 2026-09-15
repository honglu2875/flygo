"""Disabled head paths cannot influence predictions, derivatives or optimizer state."""
import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from flygo.cost import fly_cost
from flygo.fly import RustFly, PARAMETERS, packed
from flygo.readout import HeadMask, side_policy, shuffle_sides
from test_fly import fixture


def mask_fixture():
    return HeadMask(np.array([[1,0],[0,1],[1,1]],np.uint8), np.array([0,1],np.uint8),
                    json.dumps({'kind':'fixture'}))


class MaskedReadouts(unittest.TestCase):
    def test_disabled_coefficients_cannot_change_the_forward_gradient_or_clipping(self):
        graph,cfg,ports,params,*batch=fixture();mask=mask_fixture()
        perturbed={k:v.copy() for k,v in params.items()}
        for name,enabled in mask.parameter_masks().items(): perturbed[name][enabled==0]+=10000
        original=RustFly(graph,cfg,ports=ports,params=params,head_mask=mask)
        altered=RustFly(graph,cfg,ports=ports,params=perturbed,head_mask=mask)
        for step in range(3):
            for key,array in original.infer(batch[0],trace=True).items():
                np.testing.assert_array_equal(array,altered.infer(batch[0],trace=True)[key])
            losses,grad=original.loss_and_grad(*batch);other_losses,other_grad=altered.loss_and_grad(*batch)
            self.assertEqual(losses,other_losses)
            for key in PARAMETERS: np.testing.assert_array_equal(grad[key],other_grad[key])
            for name,enabled in mask.parameter_masks().items(): np.testing.assert_array_equal(grad[name][enabled==0],0)
            self.assertEqual(original.train_step(*batch,clip=.01),altered.train_step(*batch,clip=.01))
        for model,start in [(original,params),(altered,perturbed)]:
            state=model.checkpoint_arrays()
            for name,enabled in mask.parameter_masks().items():
                np.testing.assert_array_equal(state['param/'+name][enabled==0],start[name][enabled==0])
                for prefix in ('first/','second/'):np.testing.assert_array_equal(state[prefix+name][enabled==0],0)

    def test_all_enabled_mask_preserves_dense_states_gradients_and_updates_exactly(self):
        graph,cfg,ports,params,*batch=fixture()
        mask=HeadMask(np.ones((cfg.actions,cfg.groups),np.uint8),np.ones(cfg.groups,np.uint8),' { "kind": "dense" } ')
        dense=RustFly(graph,cfg,ports=ports,params=params)
        explicit=RustFly(graph,cfg,ports=ports,params=params,head_mask=mask)
        for step in range(3):
            for key,array in dense.infer(batch[0],trace=True).items():np.testing.assert_array_equal(array,explicit.infer(batch[0],trace=True)[key])
            losses,gradient=dense.loss_and_grad(*batch);other_losses,other_gradient=explicit.loss_and_grad(*batch)
            self.assertEqual(losses,other_losses)
            for key in PARAMETERS:np.testing.assert_array_equal(gradient[key],other_gradient[key])
            self.assertEqual(dense.train_step(*batch),explicit.train_step(*batch))
            for key,array in dense.checkpoint_arrays().items():np.testing.assert_array_equal(array,explicit.checkpoint_arrays()[key])

    def test_restore_requires_same_mask_provenance_and_zero_disabled_moments(self):
        graph,cfg,ports,params,*batch=fixture();mask=mask_fixture()
        model=RustFly(graph,cfg,ports=ports,params=params,head_mask=mask)
        model.train_step(*batch);saved=model.checkpoint_arrays()
        other=RustFly(graph,cfg,ports=ports,params=params,head_mask=HeadMask.from_checkpoint(saved))
        other.restore_arrays(saved)
        self.assertEqual(other.train_step(*batch),model.train_step(*batch))
        for key,array in model.checkpoint_arrays().items():np.testing.assert_array_equal(array,other.checkpoint_arrays()[key])
        current=model.checkpoint_arrays()
        for changed in [HeadMask(mask.policy[:,::-1],mask.value,mask.provenance),
                        HeadMask(mask.policy,mask.value,'{"kind":"different"}')]:
            bad={**saved,**changed.checkpoint_arrays()}
            with self.assertRaisesRegex(ValueError,'head mask or provenance'):model.restore_arrays(bad)
        with self.assertRaisesRegex(ValueError,'head mask or provenance'):
            RustFly(graph,cfg,ports=ports,params=params).restore_arrays(saved)
        for prefix in ('first/','second/'):
            bad={k:v.copy() for k,v in saved.items()};bad[prefix+'value_weight'][0]=.01
            with self.assertRaisesRegex(ValueError,'zero optimizer moments'):model.restore_arrays(bad)
            groups=[packed({name:bad[p+name] for name in PARAMETERS}) for p in ('param/','first/','second/')]
            with self.assertRaisesRegex(ValueError,'zero gradients and moments'):model.native.restore(*groups,int(saved['optimizer_step']))
        for key,array in model.checkpoint_arrays().items():np.testing.assert_array_equal(array,current[key])

    def test_independent_jax_all_derivatives_and_three_masked_adam_updates(self):
        try: import jax
        except ImportError:self.skipTest('Install the CPU JAX reference')
        from flygo.jax.model import forward,loss,adam
        jax.config.update('jax_enable_x64',True)
        graph,cfg,ports,initial,*batch=fixture();mask=mask_fixture()
        kwargs=dict(steps=cfg.steps,groups=cfg.groups,actions=cfg.actions,head_mask=mask.parameter_masks())
        model=RustFly(graph,cfg,ports=ports,params=initial,head_mask=mask)
        params=initial;first={k:np.zeros_like(v) for k,v in params.items()};second={k:np.zeros_like(v) for k,v in params.items()}
        derivative=jax.jit(jax.value_and_grad(lambda p:loss(p,graph,ports,*batch,**kwargs),has_aux=True))
        for step in range(1,4):
            expected=forward(params,graph,ports,batch[0],**kwargs)
            for key,array in model.infer(batch[0],trace=True).items():np.testing.assert_allclose(array,expected[key],rtol=2e-5,atol=2e-7)
            (_,losses),grad=derivative(params);actual_loss,actual_grad=model.loss_and_grad(*batch)
            np.testing.assert_allclose(list(actual_loss.values()),losses,atol=2e-7)
            for key in PARAMETERS:np.testing.assert_allclose(actual_grad[key],grad[key],rtol=2e-4,atol=2e-7,err_msg=key)
            params,first,second,norm=adam(params,grad,first,second,step,rate=.003,clip=.01)
            result=model.train_step(*batch,rate=.003,clip=.01)
            self.assertAlmostEqual(result['gradient_norm'],float(norm),delta=2e-7)
            state=model.checkpoint_arrays()
            for prefix,expected in [('param/',params),('first/',first),('second/',second)]:
                for key in PARAMETERS:np.testing.assert_allclose(state[prefix+key],expected[key],rtol=3e-4,atol=1e-6,err_msg=prefix+key)

    def test_anatomical_and_shuffled_masks_preserve_shared_cells_and_fan_in(self):
        sides=np.array(['L','L','R','M','L','R','R','M']);kinds=np.repeat(['a','b'],4)
        shuffled=shuffle_sides(sides,kinds,seed=19)
        for kind in np.unique(kinds):np.testing.assert_array_equal(np.sort(sides[kinds==kind]),np.sort(shuffled[kinds==kind]))
        np.testing.assert_array_equal(shuffled[sides=='M'],'M')
        actual=side_policy(sides);control=side_policy(shuffled)
        np.testing.assert_array_equal(actual.sum(axis=1),control.sum(axis=1))
        np.testing.assert_array_equal(actual[:81].reshape(9,9,-1)[:,4],1)
        np.testing.assert_array_equal(actual[-1],1)
        np.testing.assert_array_equal(actual[:,sides=='M'],1)
        np.testing.assert_array_equal(actual[:81].reshape(9,9,-1)[:,:4,sides=='R'],0)
        np.testing.assert_array_equal(actual[:81].reshape(9,9,-1)[:,5:,sides=='L'],0)
        mask=HeadMask(actual,np.ones(len(sides),np.uint8),' {"kind":"soma-side"} ')
        self.assertEqual(mask.contract,HeadMask.from_checkpoint(mask.checkpoint_arrays()).contract)
        nominal=fly_cost(100,400,20,groups=8)
        counted=fly_cost(100,400,20,groups=8,head_mask=mask)
        self.assertEqual(nominal['arithmetic_flops']-counted['arithmetic_flops'],2*(82*8-int(actual.sum())))

    def test_malformed_masks_cannot_be_silently_coerced_or_leave_empty_outputs(self):
        for policy,value in [(np.ones((3,2))*.5,[1,0]),(np.array([[1,0],[0,0],[1,1]]),[1,0]),
                             (np.ones((3,2),int),[0,0]),(np.ones((3,2),int),[1,0,1])]:
            with self.assertRaises(ValueError):HeadMask(policy,np.asarray(value),'{"kind":"fixture"}')

    def test_saved_mask_restores_through_the_go_player_without_external_mask_files(self):
        from flygo.checkpoint import save_checkpoint
        from flygo.data.loader import Sampler
        from flygo.play import load_player
        from flygo.qualify import sha256
        graph,cfg,ports,_,*_=fixture();cfg=replace(cfg,features=972,actions=82)
        mask=HeadMask(side_policy(['L','R']),np.array([0,1],np.uint8),'{"kind":"fixture"}')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'graphs/fixture';path.mkdir(parents=True)
            for key,array in graph.items():np.save(path/(key+'.npy'),array)
            graph['manifest']=dict(graph_id='fixture',arrays={key:dict(sha256=sha256(path/(key+'.npy'))) for key in graph})
            (path/'manifest.json').write_text(json.dumps(graph['manifest']))
            model=RustFly(graph,cfg,ports=ports,head_mask=mask)
            checkpoint=root/'model.npz'
            with patch('flygo.checkpoint.StorageBudget'):
                save_checkpoint(model,Sampler({},{}),checkpoint,dict(dataset_id='fixture'),root=root)
            player,_=load_player(checkpoint,path,threads=cfg.threads)
            self.assertEqual(player.head_mask.contract,mask.contract)
            batch=np.random.default_rng(1).normal(size=(2,9,9,12)).astype(np.float32)
            for key,array in model.infer(batch,trace=True).items():np.testing.assert_array_equal(array,player.infer(batch,trace=True)[key])

    def test_input_qualification_cannot_authorize_a_different_head_mask(self):
        from flygo.attachments import require_qualification
        _,cfg,*_=fixture();mask=mask_fixture();contract={'mode':'current'}
        settings=dict(batch_size=32,rate=.03,epsilon=1e-6,clip=1,rate_scales={'bias':.01})
        report=dict(status='complete',runtime_sha256={},records=[dict(input_contract=contract,
            batch_size=32,updates=3,optimizer={k:v for k,v in settings.items() if k!='batch_size'},
            model=asdict(cfg),head_mask=mask.contract)])
        with tempfile.TemporaryDirectory() as temporary,patch('flygo.attachments.runtime_hashes',return_value={}):
            path=Path(temporary)/'qualification.json';path.write_text(json.dumps(report))
            require_qualification(path,contract,cfg,**settings,head_mask=mask)
            for other in (None,HeadMask(mask.policy[:,::-1],mask.value,mask.provenance)):
                with self.assertRaisesRegex(ValueError,'No matching full-circuit'):
                    require_qualification(path,contract,cfg,**settings,head_mask=other)

    def test_jax_cpu_learner_preserves_masked_moments_and_continuation(self):
        try: import jax
        except ImportError:self.skipTest('Install the CPU JAX reference')
        from flygo.jax.learner import JaxFly
        graph,cfg,ports,params,*batch=fixture();mask=mask_fixture()
        model=JaxFly(graph,cfg,ports=ports,params=params,head_mask=mask)
        model.train_step(*batch,clip=.01)
        saved=model.checkpoint_arrays()
        other=JaxFly(graph,cfg,ports=ports,params=params,head_mask=HeadMask.from_checkpoint(saved))
        other.restore_arrays(saved)
        self.assertEqual(model.train_step(*batch,clip=.01),other.train_step(*batch,clip=.01))
        for key,array in model.checkpoint_arrays().items():np.testing.assert_array_equal(array,other.checkpoint_arrays()[key])
        for name,enabled in mask.parameter_masks().items():
            for prefix in ('first/','second/'):np.testing.assert_array_equal(model.checkpoint_arrays()[prefix+name][enabled==0],0)
        bad={k:v.copy() for k,v in saved.items()};bad['first/value_weight'][0]=.01
        with self.assertRaisesRegex(ValueError,'zero optimizer moments'):other.restore_arrays(bad)
        with self.assertRaisesRegex(ValueError,'head mask or provenance'):
            JaxFly(graph,cfg,ports=ports,params=params).restore_arrays(saved)


if __name__=='__main__':unittest.main()
