import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('audit_input_novelty',Path(__file__).parents[1]/'scripts/audit_input_novelty.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class InputNoveltyTests(unittest.TestCase):
    def test_keys_remove_only_the_declared_inputs_and_all_d4_orientations(self):
        x=np.zeros((1,9,9,12),np.float32);x[:,1,3,0]=1;x[:,2,7,1]=1
        x[:,2,4,11]=1;x[...,9]=-.125;x[...,10]=.5
        for mode in ('current','neutral'):
            key=audit.source_keys(x,mode)
            for flip in (False,True):
                for turns in range(4):
                    y=np.rot90(np.flip(x,axis=2) if flip else x,turns,axes=(1,2))
                    self.assertEqual(key,audit.source_keys(y,mode))
            y=x.copy();y[...,2:8]=1
            self.assertEqual(key,audit.source_keys(y,mode))
            for channel in (8,9,10):
                y=x.copy();y[...,channel]+=.25
                self.assertNotEqual(key,audit.source_keys(y,mode))
            y=x.copy();y[...,11]=0
            self.assertNotEqual(key,audit.source_keys(y,mode))
        y=x.copy();y[...,:2]=0
        self.assertEqual(audit.source_keys(x,'neutral'),audit.source_keys(y,'neutral'))
        self.assertNotEqual(audit.source_keys(x,'current'),audit.source_keys(y,'current'))


if __name__=='__main__':unittest.main()
