"""Policy concentration is measured on legal moves without changing primary losses."""
import unittest
import numpy as np
from scripts.compare_checkpoints import predict_metrics,summarize


class ValidationConfidence(unittest.TestCase):
    def test_uniform_legal_policies_have_known_entropy_and_peak_probability(self):
        class Model:
            def infer(self,features):
                return dict(logits=np.zeros((len(features),3),np.float32),value=np.zeros(len(features),np.float32))
        arrays=dict(features=np.zeros((2,1),np.float32),legal=np.asarray([[True,False,True],[True,True,True]]),
                    raw_policy=np.asarray([[.5,0,.5],[1/3,1/3,1/3]],np.float32),raw_value=np.zeros(2,np.float32))
        indices=np.arange(2);model=Model()
        ordinary=predict_metrics(model,arrays,indices,1)
        measured,confidence=predict_metrics(model,arrays,indices,1,confidence=True)
        np.testing.assert_array_equal(measured,ordinary)
        np.testing.assert_allclose(confidence[:,0],np.log([2,3]),rtol=1e-7)
        np.testing.assert_allclose(confidence[:,1],[.5,1/3],rtol=1e-7)
        report=summarize(confidence,np.asarray(['family','family']),names=('entropy','peak'))
        self.assertEqual(report['games'],1)
        np.testing.assert_allclose(report['peak']['game_bootstrap_95'],[confidence[:,1].mean()]*2)

    def test_metric_columns_cannot_be_silently_misnamed(self):
        with self.assertRaisesRegex(ValueError,'columns differ'):
            summarize(np.zeros((2,2)),np.arange(2))


if __name__=='__main__':unittest.main()
