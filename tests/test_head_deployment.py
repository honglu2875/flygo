"""Expensive masked trials require matching recovery evidence and paired starts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from deploy_research import head_io_files
from flygo.qualify import sha256
from flygo.schedule import Schedule


class HeadDeployment(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.plan=dict(rate=.03,epsilon=1e-6,clip=1,batch_size=32,warmup_steps=100,
                       rate_scales={'bias':.01},jobs=[])
        for arm in ('dense','value-group'):
            base=self.root/arm;base.mkdir()
            native=dict(status='complete',records=[dict(model={'seed':4},head_mask={'arm':arm},input_contract={'mode':'current'})])
            self.write(base/'numerical.json',native)
            self.write(base/'initial.json',dict(arrays={'weights':'shared'},sampler={'seed':4}))
            self.write(base/'expected.json',dict(continued=True))
            report=dict(status='passed',seed=4,runtime_sha256={},qualification_sha256=sha256(base/'numerical.json'),
                model={'seed':4},head_mask={'arm':arm},input_contract={'mode':'current'},
                training_contract=dict(batch_size=32,rate=.03,epsilon=1e-6,clip=1,rate_scales={'bias':.01},schedule=Schedule(.03,100,0,.1).contract()),
                fresh_process=dict(status='passed',runtime_sha256={}),initial_sha256=sha256(base/'initial.json'),expected_sha256=sha256(base/'expected.json'))
            self.write(base/'result.json',report)
            self.plan['jobs'].append(dict(seed=4,head_mask=arm+'.npz',qualification=arm+'/numerical.json',io_qualification=arm+'/result.json'))
        self.patcher=patch('flygo.attachments.runtime_hashes',return_value={});self.patcher.start();self.addCleanup(self.patcher.stop)

    @staticmethod
    def write(path,value):path.write_text(json.dumps(value))

    def test_valid_evidence_cannot_cover_a_different_schedule_or_missing_recovery(self):
        self.assertEqual(len(head_io_files(self.plan,self.root)),6)
        self.plan['warmup_steps']=0
        with self.assertRaisesRegex(ValueError,'trial contract'):head_io_files(self.plan,self.root)
        self.plan['warmup_steps']=100;self.plan['jobs'][0].pop('io_qualification')
        with self.assertRaisesRegex(ValueError,'actual IO'):head_io_files(self.plan,self.root)

    def test_a_valid_but_unpaired_initial_state_is_rejected(self):
        base=self.root/'value-group';self.write(base/'initial.json',dict(arrays={'weights':'different'},sampler={'seed':4}))
        report=json.loads((base/'result.json').read_text());report['initial_sha256']=sha256(base/'initial.json');self.write(base/'result.json',report)
        with self.assertRaisesRegex(ValueError,'different initial'):head_io_files(self.plan,self.root)

    def test_changed_next_update_or_mask_evidence_is_rejected(self):
        base=self.root/'dense';self.write(base/'expected.json',dict(continued=False))
        with self.assertRaisesRegex(ValueError,'evidence changed'):head_io_files(self.plan,self.root)
        self.write(base/'expected.json',dict(continued=True))
        report=json.loads((base/'result.json').read_text());report['head_mask']={'arm':'different'};self.write(base/'result.json',report)
        with self.assertRaisesRegex(ValueError,'trial contract'):head_io_files(self.plan,self.root)


if __name__=='__main__':unittest.main()
