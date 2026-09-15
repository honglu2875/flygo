"""Causal image ablations and a fixed, explicit context route."""
import unittest
import json
from dataclasses import asdict
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from flygo.attachments import VERSION, VisualContext, context_ports, input_contract
from flygo.vision import SphericalRenderer, directions


class AttachmentTests(unittest.TestCase):
    def setUp(self):
        u = directions(np.deg2rad([-25]*4),np.deg2rad([-30,30,-30,30]))
        self.adapter = VisualContext(SphericalRenderer.build(u,np.array(['L','L','R','R'])))
        self.x = np.zeros((2,9,9,12),np.float32)
        self.x[...,8] = 1
        self.x[...,9] = -.1
        self.x[...,10] = .5
        self.x[:,1,3,11] = 1

    def test_current_board_is_coherent_and_has_no_history_leak(self):
        x = self.x.copy(); x[:,4,4,0] = 1
        current = self.adapter.encode(x,'current')
        np.testing.assert_array_equal(current[:,:4],np.repeat(current[:,:1],4,axis=1))
        changed = x.copy(); changed[...,2:8] = np.random.default_rng(1).integers(0,2,changed[...,2:8].shape)
        np.testing.assert_array_equal(current,self.adapter.encode(changed,'current'))
        self.assertFalse(np.array_equal(self.adapter.encode(x,'history')[:,:4],current[:,:4]))
        np.testing.assert_array_equal(x[...,:2],changed[...,:2])

    def test_visual_and_context_interventions_are_separate(self):
        reference = self.adapter.encode(self.x,'history')
        changed = self.x.copy(); changed[...,9] = .2; changed[:,1,3,11] = 0
        encoded = self.adapter.encode(changed,'history')
        np.testing.assert_array_equal(reference[:,:4],encoded[:,:4])
        self.assertEqual(reference[0,4+12],1)
        np.testing.assert_array_equal(reference[:,-3:],self.x[:,0,0,8:11])
        for mode in ('current','neutral'):
            np.testing.assert_array_equal(reference[:,4:],self.adapter.encode(self.x,mode)[:,4:])
        np.testing.assert_array_equal(self.adapter.encode(changed,'neutral')[:,:4],np.full((2,4),.5,np.float32))
        changed[:,0,0,9] = .3
        with self.assertRaisesRegex(ValueError,'spatially constant'):self.adapter.encode(changed,'history')

    def test_context_assignment_preserves_neural_identities(self):
        ports = context_ports(100,[1,2],[98,99],np.arange(3,98))
        np.testing.assert_array_equal(ports['input_index'][[1,2]],[0,1])
        np.testing.assert_array_equal(np.unique(ports['input_index'][3:98]),np.arange(2,86))
        np.testing.assert_array_equal(ports['output_group'][[98,99]],[0,1])
        np.testing.assert_array_equal(ports['input_index'],context_ports(100,[1,2],[98,99],np.arange(3,98))['input_index'])
        with self.assertRaises(ValueError):context_ports(100,[1,2],[98,99],np.arange(2,98))

    def test_one_large_board_is_bilateral_local_and_independent_of_history(self):
        row,col=np.meshgrid(np.linspace(55,-55,9),np.linspace(-50,0,9),indexing='ij')
        points=directions(np.deg2rad(col.ravel()),np.deg2rad(row.ravel()))
        unit=np.tile(points,(2,1));side=np.repeat(['L','R'],81)
        renderer=SphericalRenderer.build(unit,side,patch_centers_degrees=[[-25,0]],
            halfwidth_degrees=[25,55],patch_lags=[[0,0]])
        adapter=VisualContext(renderer)
        x=self.x.copy();x[:,4,4,0]=1
        encoded=adapter.encode(x,'current')
        np.testing.assert_array_equal(encoded[:,:81],encoded[:,81:162])
        altered=x.copy();altered[...,2:8]=1
        np.testing.assert_array_equal(encoded,adapter.encode(altered,'current'))
        np.testing.assert_array_equal(encoded,adapter.encode(altered,'history'))
        distance=np.arccos(np.clip(unit@points[40],-1,1))
        self.assertTrue(np.all(encoded[:,:162][:,distance>np.deg2rad(10)+1e-12]==.5))
        self.assertTrue(np.all(encoded[:,[40,121]]>.5))
        np.testing.assert_array_equal(adapter.encode(x,'neutral')[:,:162],.5)
        self.assertEqual([row['covered_board_points'] for row in renderer.audit()],[81,81])
        self.assertEqual([row['lag'] for row in renderer.audit()],[0,0])
        for kwargs in (dict(patch_lags=[[0,4]]),dict(patch_lags=[[0.,0.]]),
                       dict(halfwidth_degrees=[25,90]),dict(patch_lags=[[0,1],[2,3]])):
            spec=dict(patch_centers_degrees=[[-25,0]],halfwidth_degrees=[25,55],patch_lags=[[0,0]])
            with self.assertRaises(ValueError):SphericalRenderer.build(unit,side,**(spec|kwargs))

    def test_training_qualification_must_match_the_actual_head_and_sampler_seed(self):
        from dataclasses import replace
        from flygo.attachments import require_qualification
        from flygo.fly import FlyConfig
        config=FlyConfig(seed=2)
        contract=dict(version=VERSION,attachment_sha256='fixture',mode='current')
        optimizer=dict(batch_size=32,rate=.03,epsilon=1e-6,clip=1,rate_scales={'bias':.01})
        report=dict(status='complete',runtime_sha256={},records=[dict(input_contract=contract,
            batch_size=32,updates=3,optimizer={k:v for k,v in optimizer.items() if k!='batch_size'},
            model=asdict(config))])
        with tempfile.TemporaryDirectory() as temp,patch('flygo.attachments.runtime_hashes',return_value={}):
            path=Path(temp)/'qualification.json';path.write_text(json.dumps(report))
            require_qualification(path,contract,config,**optimizer)
            with self.assertRaisesRegex(ValueError,'No matching full-circuit'):
                require_qualification(path,contract,replace(config,seed=3),**optimizer)

    def test_checkpoint_player_restores_the_external_encoding(self):
        from flygo.fly import FlyConfig,RustFly
        from flygo.checkpoint import save_checkpoint
        from flygo.data.loader import Sampler
        from flygo.play import load_player
        from flygo.qualify import sha256
        sensors=np.arange(1,5);motors=np.array([98,99]);context=np.arange(5,98)
        ports=context_ports(100,sensors,motors,context)
        graph=dict(src=np.array([1,3,5,6],np.int32),dst=np.array([98,98,99,99],np.int32),
            indptr=np.r_[np.zeros(99,np.int32),2,4].astype(np.int32),
            type_id=np.zeros(100,np.int32),sign=np.ones(100,np.float32),
            strength=np.full(4,.1,np.float32),sensory=np.arange(1,98,dtype=np.int32))
        config=FlyConfig(steps=2,features=88,groups=2,threads=2)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);directory=root/'graphs/fixture';directory.mkdir(parents=True)
            for k,a in graph.items():np.save(directory/(k+'.npy'),a)
            graph['manifest']=dict(graph_id='fixture',arrays={k:dict(sha256=sha256(directory/(k+'.npy'))) for k in graph})
            (directory/'manifest.json').write_text(json.dumps(graph['manifest']))
            artifact=root/'ports/fixture/attachment.npz';artifact.parent.mkdir(parents=True)
            np.savez(artifact,**self.adapter.renderer.arrays(),sensors=sensors,motors=motors,
                     context_nodes=context,**ports)
            receipt=dict(version=VERSION,status='complete',sha256=sha256(artifact),graph_id='fixture',
                         dataset_id='data',features=88,groups=2,context_seed=918421)
            artifact.with_suffix('.json').write_text(json.dumps(receipt))
            core=RustFly(graph,config,ports=ports)
            checkpoint=root/'checkpoint.npz'
            with patch('flygo.checkpoint.StorageBudget'):
                save_checkpoint(core,Sampler({},{}),checkpoint,dict(dataset_id='data',
                    input_contract=input_contract(receipt,'current')),root=root)
            player,metadata=load_player(checkpoint,directory,threads=2)
            x=self.x.copy();x[:,4,4,0]=1
            for key,a in core.infer(self.adapter.encode(x,'current'),trace=True).items():
                np.testing.assert_array_equal(a,player.infer(x,trace=True)[key])
            self.assertEqual(metadata['input_contract']['mode'],'current')
            self.assertEqual(asdict(player.config),asdict(config))
            artifact.with_suffix('.json').unlink()
            with self.assertRaisesRegex(ValueError,'Missing qualified visual input map'):
                load_player(checkpoint,directory,threads=2)


if __name__ == '__main__': unittest.main()
