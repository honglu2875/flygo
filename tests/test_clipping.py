"""Independent optimizer math, truthful diagnostics and portable clipping contracts."""
import contextlib
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from flygo.fly import RustFly
from flygo.optimizer import (check_clipping_restore, check_clipping_resume,
                             clipping_metrics, saved_clipping_mode)
from test_fly import fixture


class Clipping(unittest.TestCase):
    def test_first_group_update_matches_independent_adam_and_actual_factors(self):
        graph,config,ports,params,*batch=fixture()
        model=RustFly(graph,config,ports=ports,params=params,clip_mode='parameter-group')
        _,gradient=model.loss_and_grad(*batch)
        clip=.01;epsilon=1e-4;rate=.003
        metrics=model.train_step(*batch,clip=clip,epsilon=epsilon,rate=rate,rate_scales={'bias':.01})
        for name,g32 in gradient.items():
            g=g32.astype(float);norm=np.linalg.norm(g)
            factor=min(1.,clip/max(norm,1e-30));clipped=g*factor
            step_rate=rate*(.01 if name=='bias' else 1)
            expected=params[name]-step_rate*clipped/(np.abs(clipped)+epsilon)
            np.testing.assert_allclose(model.parameters()[name],expected,rtol=2e-6,atol=2e-7,err_msg=name)
            self.assertAlmostEqual(metrics['clipping']['group_norms'][name],norm,places=12)
            self.assertAlmostEqual(metrics['clipping']['factors'][name],factor,places=6)
        self.assertEqual(metrics['clipping']['mode'],'parameter-group')
        self.assertTrue(metrics['clipping']['clipped'])

    def test_default_mode_and_explicit_global_are_identical(self):
        graph,config,ports,params,*batch=fixture()
        left=RustFly(graph,config,ports=ports,params=params)
        right=RustFly(graph,config,ports=ports,params=params,clip_mode='global')
        for _ in range(3):
            self.assertEqual(left.train_step(*batch,clip=.01),right.train_step(*batch,clip=.01))
        for key,value in left.checkpoint_arrays().items():
            np.testing.assert_array_equal(value,right.checkpoint_arrays()[key])
        self.assertNotIn('optimizer_clip_mode',left.checkpoint_arrays())
        with self.assertRaises(AttributeError):left.clip_mode='parameter-group'

    def test_mode_tags_and_wrong_restores_are_atomic(self):
        graph,config,ports,params,*batch=fixture()
        group=RustFly(graph,config,ports=ports,params=params,clip_mode='parameter-group')
        legacy=RustFly(graph,config,ports=ports,params=params)
        for model,wrong in [(group,legacy.checkpoint_arrays()),(legacy,group.checkpoint_arrays())]:
            before=model.checkpoint_arrays()
            with self.assertRaisesRegex(ValueError,'clipping mode differs'):model.restore_arrays(wrong)
            for key,value in before.items():np.testing.assert_array_equal(value,model.checkpoint_arrays()[key])
        for tag in [np.asarray(2,np.uint8),np.asarray(1.),np.asarray([1],np.uint8)]:
            with self.assertRaisesRegex(ValueError,'Invalid checkpoint'):
                check_clipping_restore('parameter-group',{'optimizer_clip_mode':tag})
        check_clipping_resume('global',{})
        with self.assertRaisesRegex(ValueError,'differs'):check_clipping_resume('parameter-group',{})
        with self.assertRaisesRegex(ValueError,'Inconsistent'):
            saved_clipping_mode({'optimizer_clip_mode':'global','training_contract':{'clip_mode':'parameter-group'}})

    def test_checkpoint_binds_mode_and_next_update(self):
        from flygo.checkpoint import save_checkpoint,load_checkpoint
        from flygo.data.loader import Sampler
        graph,config,ports,params,*batch=fixture();graph['manifest']={'graph_id':'fixture'}
        model=RustFly(graph,config,ports=ports,params=params,clip_mode='parameter-group')
        sampler=Sampler({}, {}, 13);model.train_step(*batch,clip=.01)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'checkpoint.npz'
            with patch('flygo.checkpoint.StorageBudget.reserve',return_value=contextlib.nullcontext()):
                save_checkpoint(model,sampler,path,{'dataset_id':'fixture'},root=root)
            expected=model.train_step(*batch,clip=.01);arrays=model.checkpoint_arrays()
            restored=RustFly(graph,config,ports=ports,params=params,clip_mode='parameter-group')
            info=load_checkpoint(path,restored,sampler,dataset_id='fixture')
            self.assertEqual(info['optimizer_version'],'adam-fp32-group-clipping-v1')
            self.assertEqual(restored.train_step(*batch,clip=.01),expected)
            for key,value in arrays.items():np.testing.assert_array_equal(value,restored.checkpoint_arrays()[key])
            wrong=RustFly(graph,config,ports=ports,params=params)
            with self.assertRaisesRegex(ValueError,'clipping mode differs'):load_checkpoint(path,wrong)

    def test_qualification_cannot_substitute_global_for_group_clipping(self):
        from flygo.attachments import require_qualification,runtime_hashes
        _,config,*_=fixture();contract={'mode':'current'}
        optimizer=dict(rate=.03,epsilon=1e-6,clip=1.,rate_scales={'bias':.01})
        record=dict(input_contract=contract,head_mask=None,batch_size=2,updates=3,model=asdict(config),optimizer=optimizer)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'qualification.json'
            path.write_text(json.dumps(dict(status='complete',runtime_sha256=runtime_hashes(),records=[record])))
            require_qualification(path,contract,config,batch_size=2,**optimizer)
            with self.assertRaisesRegex(ValueError,'No matching'):
                require_qualification(path,contract,config,batch_size=2,clip_mode='parameter-group',**optimizer)

    def test_metrics_use_applied_group_factors(self):
        result=clipping_metrics('parameter-group',{'a':.8,'b':.8},{'a':1.,'b':1.})
        self.assertGreater(np.linalg.norm(list(result['group_norms'].values())),1.)
        self.assertFalse(result['clipped'])

    def test_group_rust_jax_updates_and_restoration(self):
        try:import jax
        except ImportError:self.skipTest('JAX optional dependency')
        from flygo.jax.learner import JaxFly
        graph,config,ports,params,*small=fixture()
        batch=[np.concatenate([a]*jax.device_count()) for a in small]
        rust=RustFly(graph,config,ports=ports,params=params,clip_mode='parameter-group')
        reference=JaxFly(graph,config,ports=ports,params=params,clip_mode='parameter-group')
        settings=dict(rate=.003,clip=.01,epsilon=1e-6,rate_scales={'bias':.01})
        for _ in range(3):
            left=rust.train_step(*batch,**settings);right=reference.train_step(*batch,**settings)
            for key in ['group_norms','factors']:
                for name,value in left['clipping'][key].items():
                    np.testing.assert_allclose(right['clipping'][key][name],value,rtol=3e-4,atol=2e-6)
            for key,value in rust.checkpoint_arrays().items():
                np.testing.assert_allclose(reference.checkpoint_arrays()[key],value,rtol=3e-4,atol=2e-6,err_msg=key)
        for model in [rust,reference]:
            saved=model.checkpoint_arrays();model.train_step(*batch,**settings);expected=model.checkpoint_arrays()
            model.restore_arrays(saved);model.train_step(*batch,**settings)
            for key,value in expected.items():np.testing.assert_array_equal(value,model.checkpoint_arrays()[key])


if __name__=='__main__':unittest.main()
