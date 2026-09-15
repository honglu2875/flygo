import unittest
import numpy as np
try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None
from flygo.readout_convex import decompose,condition,coefficients,original_gradient,RidgePolicyObjective
from flygo.readout_probe import loss_and_grad,forward


@unittest.skipIf(minimize is None, 'Install the optional research extra for SciPy')
class ConvexReadoutTests(unittest.TestCase):
    def fixture(self, columns=7):
        rng=np.random.default_rng(172)
        x=rng.normal(size=(29,columns))
        if columns:x[:,-1]=3  # No training variation; keep this coordinate.
        base=rng.normal(size=(29,5));legal=rng.uniform(size=base.shape)>.2;legal[:,0]=True
        q=rng.uniform(size=base.shape)*legal;q=q/q.sum(axis=1,keepdims=True)*.9999999
        mean,values,vectors=decompose(x);ridge=.03
        white=condition(x,mean,values,vectors,ridge)
        objective=RidgePolicyObjective(white,base,legal,q,values,ridge)
        return rng,x,base,legal,q,mean,values,vectors,ridge,objective

    def test_basis_preserves_logits_regularization_and_gradient(self):
        rng,x,base,legal,q,mean,d,v,ridge,obj=self.fixture()
        matrix=rng.normal(0,.1,obj.shape)
        params=coefficients(matrix,d,v,ridge)
        raw_logits=forward(params,x-mean,base)
        np.testing.assert_allclose(raw_logits,base+obj.features@matrix[:-1]+matrix[-1],rtol=0,atol=1e-14)
        loss,g=obj(matrix.ravel());ce,raw=loss_and_grad(params,x-mean,base,legal,q)
        self.assertAlmostEqual(loss,ce+ridge/2*sum(np.sum(a*a) for a in params.values()),places=13)
        expected=np.concatenate((raw['weight']+ridge*params['weight'],(raw['bias']+ridge*params['bias'])[None]))
        np.testing.assert_allclose(original_gradient(g.reshape(obj.shape),d,v,ridge),expected,rtol=1e-12,atol=1e-13)

    def test_full_objective_directional_derivative(self):
        rng,*rest,obj=self.fixture()
        vector=rng.normal(0,.1,np.prod(obj.shape));direction=rng.normal(size=len(vector));h=1e-6
        _,g=obj(vector)
        numeric=(obj(vector+h*direction)[0]-obj(vector-h*direction)[0])/(2*h)
        self.assertAlmostEqual(numeric,float(g@direction),places=8)

    def test_stationarity_bound_contains_known_optimum_gap(self):
        rng=np.random.default_rng(18);x=rng.normal(size=(31,6));mean,d,v=decompose(x);ridge=.01
        base=np.zeros((31,4));q=np.full_like(base,.25);legal=np.ones_like(base,bool)
        obj=RidgePolicyObjective(condition(x,mean,d,v,ridge),base,legal,q,d,ridge)
        matrix=rng.normal(0,.1,obj.shape);loss,g=obj(matrix.ravel())
        raw=original_gradient(g.reshape(obj.shape),d,v,ridge)
        self.assertLessEqual(loss-np.log(4),float(np.sum(raw*raw)/(2*ridge))+1e-12)
        result=minimize(obj,matrix.ravel(),jac=True,method='L-BFGS-B',options=dict(gtol=1e-10,ftol=1e-14))
        self.assertTrue(result.success);self.assertAlmostEqual(result.fun,np.log(4),places=11)

    def test_bias_only_fit_and_strictly_positive_ridge(self):
        _,x,base,legal,q,mean,d,v,ridge,obj=self.fixture(columns=0)
        self.assertEqual(obj.shape,(1,5))
        result=minimize(obj,np.zeros(5),jac=True,method='L-BFGS-B',options=dict(gtol=1e-9,ftol=1e-14))
        self.assertTrue(result.success)
        self.assertLess(result.fun,obj(np.zeros(5))[0])
        with self.assertRaises(ValueError):condition(x,mean,d,v,0)


if __name__=='__main__':unittest.main()
