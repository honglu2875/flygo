"""Independent convolution arithmetic and a learnable, portable neural control."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np


class CNN(unittest.TestCase):
    def setUp(self):
        try:import jax
        except ImportError:self.skipTest('JAX optional dependency')

    def test_convolution_padding_and_channel_order(self):
        from flygo.jax.cnn import convolution
        rng=np.random.default_rng(205)
        x=rng.normal(size=(2,5,4,3)).astype(np.float32)
        kernel=rng.normal(size=(3,3,3,2)).astype(np.float32)
        bias=rng.normal(size=2).astype(np.float32)
        padded=np.pad(x,((0,0),(1,1),(1,1),(0,0)))
        expected=np.empty((2,5,4,2))
        for i in range(5):
            for j in range(4):
                expected[:,i,j]=np.einsum('bxyc,xyco->bo',padded[:,i:i+3,j:j+3].astype(float),kernel.astype(float))+bias
        np.testing.assert_allclose(convolution(x,kernel,bias),expected,rtol=2e-5,atol=4e-6)

    def test_learning_checkpoint_and_gtp(self):
        import jax
        from flygo.jax.cnn import CNNConfig,JaxCNN
        from flygo.checkpoint import save_checkpoint
        from flygo.data.loader import Sampler
        from flygo.play import load_player,choose_move
        from flygo.go import Game,GameConfig
        model=JaxCNN(CNNConfig(channels=4,blocks=1,threads=1))
        count=2*jax.device_count()
        rng=np.random.default_rng(941)
        x=(rng.random((count,9,9,12))<.2).astype(np.float32)
        legal=np.ones((count,82),bool);legal[:,1]=False
        policy=np.zeros((count,82),np.float32);policy[:,40]=1
        value=np.full(count,.25,np.float32)
        first=model.train_step(x,legal,policy,value,rate=.005)
        for _ in range(39):last=model.train_step(x,legal,policy,value,rate=.005)
        self.assertLess(last['policy_loss']+last['value_loss'],.5*(first['policy_loss']+first['value_loss']))
        saved=model.checkpoint_arrays()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'control.npz'
            with patch('flygo.checkpoint.StorageBudget'):
                save_checkpoint(model,Sampler({},{}),path,dict(dataset_id='fixture'),root=Path(directory))
            restored,metadata=load_player(path,Path('/unused'),threads=1)
            self.assertEqual(metadata['model_version'],'residual-cnn-v1')
            for name,want in saved.items():np.testing.assert_array_equal(restored.checkpoint_arrays()[name],want)
            game=Game(GameConfig(simulations=0))
            move=choose_move(game,restored)
            game.play(game.state().to_play,move.action)

    def test_declared_compute_budget(self):
        from flygo.cost import fly_cost,cnn_cost
        fly=fly_cost(165122,15270273,15912)['arithmetic_flops']
        cnn=cnn_cost()['arithmetic_flops']
        self.assertLess(abs(cnn/fly-1),.01)


if __name__=='__main__':unittest.main()
