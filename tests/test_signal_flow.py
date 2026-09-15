"""Independent chain-rule and directed-path checks for propagation diagnostics."""
import unittest
import numpy as np

try:
    from flygo.signal_flow import Linearization, distances, node_signal, pooled_cotangent
except ImportError:
    Linearization = None


@unittest.skipIf(Linearization is None, 'Install the optional SciPy research extra')
class SignalFlowTests(unittest.TestCase):
    def fixture(self):
        graph = dict(indptr=np.array([0,1,3,4]), src=np.array([1,0,2,1]), dst=np.array([0,1,1,2]),
                     type_id=np.array([0,1,0]), sign=np.array([1.,-1.,1.], np.float32))
        params = dict(edge=np.array([-1.,-.3,-.7,-1.4],np.float32),
                      leak=np.array([.3,-.4],np.float32), bias=np.array([.4,.3],np.float32))
        linear = Linearization(graph,params)
        return graph,params,linear

    @staticmethod
    def forward(linear, bias, drive, start):
        states=[start]
        for _ in range(4):
            states.append((1-linear.alpha[:,None])*states[-1]+linear.alpha[:,None]*(
                linear.matrix@np.maximum(states[-1],0)+bias[linear.types,None]+drive))
        return states

    def test_adjoint_drive_and_initial_state_match_directional_differences(self):
        _,params,linear=self.fixture();rng=np.random.default_rng(4)
        drive=rng.uniform(.1,.3,(3,2));initial=np.full((3,2),.07);last=rng.normal(size=(3,2))
        states=self.forward(linear,params['bias'],drive,initial)
        _,gradient,cotangents=linear.adjoint(states,last,edge_chunk=2)
        direction=rng.normal(size=drive.shape);h=1e-6
        def value(d,s):return np.sum(self.forward(linear,params['bias'],d,s)[-1]*last)
        numeric=(value(drive+h*direction,initial)-value(drive-h*direction,initial))/(2*h)
        self.assertAlmostEqual(numeric,float(np.sum(gradient*direction)),places=8)
        numeric=(value(drive,initial+h*direction)-value(drive,initial-h*direction))/(2*h)
        self.assertAlmostEqual(numeric,float(np.sum(cotangents[0]*direction)),places=8)

    def test_edge_and_type_bias_gradients_match_real_valued_surrogate(self):
        graph,params,linear=self.fixture();rng=np.random.default_rng(51)
        drive=rng.uniform(.1,.3,(3,2));initial=np.full((3,2),.07);last=rng.normal(size=(3,2))
        states=self.forward(linear,params['bias'],drive,initial)
        gradient,_,_=linear.adjoint(states,last,edge_chunk=2)
        h=1e-6;direction=rng.normal(size=len(graph['src']))
        values=[]
        for shift in (-h,h):
            altered=Linearization(graph,params)
            altered.matrix.data += shift*direction*linear.edge_slope
            values.append(np.sum(self.forward(altered,params['bias'],drive,initial)[-1]*last))
        self.assertAlmostEqual((values[1]-values[0])/(2*h),float(gradient['edge']@direction),places=8)
        direction=rng.normal(size=2)
        values=[np.sum(self.forward(linear,params['bias'].astype(float)+shift*direction,drive,initial)[-1]*last) for shift in (-h,h)]
        self.assertAlmostEqual((values[1]-values[0])/(2*h),float(gradient['bias']@direction),places=8)
        values=[]
        for shift in (-h,h):
            altered=Linearization(graph,params)
            altered.alpha += shift*direction[linear.types]*linear.leak_slope
            values.append(np.sum(self.forward(altered,params['bias'],drive,initial)[-1]*last))
        self.assertAlmostEqual((values[1]-values[0])/(2*h),float(gradient['leak']@direction),places=8)

    def test_directed_distance_horizon_and_context_cancelation(self):
        # 0 -> 1 -> 2, plus isolated 3.
        ptr=np.array([0,0,1,2,2]);src=np.array([0,1])
        np.testing.assert_array_equal(distances(ptr,src,[0],1),[0,1,-1,-1])
        np.testing.assert_array_equal(distances(ptr,src,[2],2,reverse=True),[2,1,0,-1])
        current=np.array([[2.,4.],[1.,1.]])+10
        neutral=current-np.array([[1.,3.],[0.,0.]])
        result=node_signal(current,neutral)
        np.testing.assert_array_equal(result['visual_std'],[1.,0.])
        np.testing.assert_array_equal(result['visual_mean'],[2.,0.])

    def test_original_and_fitted_policy_cotangents_match_local_loss(self):
        rng=np.random.default_rng(916);batch,motors,actions=4,5,7
        raw=rng.uniform(.2,1.,(batch,motors));gain=rng.uniform(.7,1.3,motors)
        weights=rng.normal(size=(actions,motors));q=rng.uniform(size=(batch,actions))
        legal=rng.uniform(size=q.shape)>.25;legal[:,0]=True;q*=legal;q=q/q.sum(axis=1,keepdims=True)*.9999999
        params=dict(policy_weight=weights.ravel(),readout_gain=gain)
        residual=dict(legal=legal,weight=rng.normal(size=(motors,actions)),bias=rng.normal(size=actions),
                      mean=np.zeros(motors),scale=rng.uniform(.4,1.,motors),center=np.zeros(motors))
        direction=rng.normal(size=raw.shape);h=1e-6
        for name in ('original-policy','fitted-linear-policy'):
            def result(pooled):
                response=pooled/gain
                output=dict(logits=pooled@weights.T,states=[response.T])
                return pooled_cotangent(name,output,params,np.arange(motors),q,np.zeros(batch),residual)
            extra,_,_,metrics=result(raw*gain)
            values=[result(raw*gain+shift*direction)[3]['policy_kl'] for shift in (-h,h)]
            self.assertAlmostEqual((values[1]-values[0])/(2*h),float(np.sum(extra*direction)),places=6)

    def test_value_cotangent_matches_tanh_mean_square_loss(self):
        rng=np.random.default_rng(215);pooled=rng.normal(size=(4,5));weight=rng.normal(size=5)
        target=rng.uniform(-1,1,4);direction=rng.normal(size=pooled.shape);h=1e-6
        def loss(x):return np.mean((np.tanh(x@weight)-target)**2)
        output=dict(logits=np.zeros((4,7)),value=np.tanh(pooled@weight))
        params=dict(policy_weight=np.zeros(35),readout_gain=np.ones(5),value_weight=weight)
        extra,score,cotangent,_=pooled_cotangent('original-value',output,params,np.arange(5),np.zeros((4,7)),target,{})
        np.testing.assert_array_equal(extra,np.zeros_like(extra))
        numeric=(loss(pooled+h*direction)-loss(pooled-h*direction))/(2*h)
        self.assertAlmostEqual(numeric,float(np.sum(cotangent*direction)),places=6)


if __name__=='__main__':unittest.main()
