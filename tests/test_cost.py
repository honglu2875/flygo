"""Executed sparse work changes with batch unions and the dense-path threshold."""
import unittest
import numpy as np

from scripts.count_inference import count_edges


class CountedWork(unittest.TestCase):
    def test_batch_union_and_dense_fallback(self):
        degree=np.array([2,3,5,7,11])
        initial=np.full((5,2),.01)
        state=np.array([[1,0],[-1,0],[0,1],[0,-1],[1,0]])
        end=np.zeros_like(state)
        counts=count_edges([initial,state,end],degree)
        self.assertEqual([r['edge_multiply_adds_per_position'] for r in counts],[28,18])
        one=count_edges([state[:,:1],end[:,:1]],degree)
        self.assertEqual(one[0]['edge_multiply_adds_per_position'],13)
        # Four of five active rows uses the dense path, even though the fifth
        # row has eleven edges that would otherwise contribute exact zeros.
        boundary=np.array([[1],[1],[1],[1],[0]])
        dense=count_edges([boundary,boundary],degree)[0]
        self.assertFalse(dense['skip_zero_rows'])
        self.assertEqual(dense['edge_multiply_adds_per_position'],28)
        self.assertEqual(count_edges([end,end],degree)[0]['edge_multiply_adds_per_position'],0)


if __name__=='__main__':unittest.main()
