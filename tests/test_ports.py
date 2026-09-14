"""Inferred column provenance, missingness and matched spatial controls."""
import unittest
import numpy as np
from flygo.ports import infer_columns,retinal_overlay


class Ports(unittest.TestCase):
    def test_column_votes_use_counts_side_and_explicit_ties(self):
        coord=np.array([[np.nan,np.nan]]*3+[[0,0],[1,0],[2,0]],float)
        side=np.array(['L','L','L','L','L','R'])
        src=np.array([0,0,0,1,1,2]);dst=np.array([3,4,5,3,4,5])
        result=infer_columns(src,dst,np.array([10,2,100,3,3,8]),coord,side,np.arange(3),side)
        np.testing.assert_array_equal(result['coordinates'][0],[0,0])
        self.assertAlmostEqual(result['confidence'][0],10/12)
        self.assertEqual(result['synapses'][0],12)
        self.assertEqual(result['side_rejected'][0],1)
        self.assertTrue(np.isnan(result['coordinates'][1:3]).all())

    def test_overlay_preserves_every_control_stratum_and_other_ports(self):
        n=12*81
        base=dict(input_index=np.arange(n,dtype=np.int32),output_group=np.full(n,-1,np.int32),
                  output_scale=np.zeros(n,np.float32))
        coordinate=np.stack([(np.arange(n)//12)%9,(np.arange(n)//108)%9],axis=1).astype(float)
        inference=dict(coordinates=coordinate,confidence=np.ones(n),synapses=np.full(n,10))
        photos=np.arange(n-12);side=np.full(n,'L');types=np.zeros(n,np.int32)
        a,b,report=retinal_overlay(base,inference,photos,side,types,seed=1)
        self.assertGreater(np.count_nonzero(a['input_index']!=b['input_index']),n//2)
        for channel in range(12):
            cells=photos[photos%12==channel]
            np.testing.assert_array_equal(np.sort(a['input_index'][cells]),np.sort(b['input_index'][cells]))
        np.testing.assert_array_equal(a['input_index'][-12:],base['input_index'][-12:])
        for ports in (a,b):
            for key in ('output_group','output_scale'):np.testing.assert_array_equal(ports[key],base[key])


if __name__=='__main__':unittest.main()
