#!/usr/bin/env python3
"""Trace visual variation and independently audit policy/value gradient flow."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy

from flygo.attachments import runtime_hashes
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release
from flygo.play import load_player
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB, StorageBudget

try:
    import signal_flow as flow
except ModuleNotFoundError:
    from flygo import signal_flow as flow


def read(path):
    return json.loads(path.read_text())


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def compare(reference, native, tolerance):
    difference = np.asarray(reference, np.float64)-np.asarray(native, np.float64)
    threshold = tolerance['atol'] + tolerance['rtol']*np.abs(native)
    return dict(passed=bool(np.all(np.abs(difference) <= threshold)),
                max_abs=float(np.max(np.abs(difference), initial=0)),
                relative_l2=float(np.linalg.norm(difference)/max(np.linalg.norm(native.astype(float)), 1e-30)),
                mismatches=int(np.count_nonzero(np.abs(difference) > threshold)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root;out=args.output;plan=read(args.plan)
    pin(plan['cpus']);out.resolve().relative_to(root.resolve())
    case=next(c for c in plan['cases'] if c['seed']==args.seed)
    bank_dir=root/plan['source_bank']/f'seed-{args.seed}'
    bank=read(bank_dir/'bank.json')
    if (sha256(bank_dir/'bank.json')!=case['bank_sha256'] or bank['runtime_sha256']!=runtime_hashes()
            or sha256(bank_dir.parent/'plan.json')!=plan['source_plan_sha256']):
        raise ValueError('Frozen bank or scientific implementation differs')
    if sha256(bank_dir/'train-labels.npz')!=bank['selection']['train']['labels_sha256']:
        raise ValueError('Training labels changed')
    if (sha256(bank_dir/'normalization.npz')!=bank['normalization_sha256']
            or sha256(bank_dir/'train-motors.npy')!=bank['selection']['train']['motors_sha256']):
        raise ValueError('Training motor bank or statistics changed')
    count=plan['forward_positions'];batch=plan['batch_size'];gradient_count=plan['gradient_positions']
    if gradient_count!=batch or count%batch:
        raise ValueError('This audit uses one gradient batch and whole forward batches')
    with StorageBudget(root).reserve(files=512<<20,heap=12*GIB,purpose='native signal and independent gradient audit'):
        out.mkdir(parents=True,exist_ok=False);started=time.time()
        manifest,arrays,indexes=load_release(root,root/'releases'/plan['release']/'manifest.json',cache=True)
        with np.load(bank_dir/'train-labels.npz',allow_pickle=False) as data:
            labels={key:data[key][:count].copy() for key in ('indices','symmetries','policy','legal','target_value','baseline')}
            expected_motors=data['motors'].copy()
        if manifest['dataset_id']!=bank['dataset_id'] or not np.isin(labels['indices'],indexes['train']).all():
            raise ValueError('Probe must use only this release\'s training positions')
        features=arrays['features'][labels['indices']].copy()
        for b,symmetry in enumerate(labels['symmetries']):
            board=features[b]
            if int(symmetry)//4:board=np.flip(board,axis=1)
            features[b]=np.rot90(board,int(symmetry)%4,axes=(0,1))
        np.savez(out/'selection.npz',**labels,features=features,motors=expected_motors)
        del arrays,indexes
        graph_path=Path(read(root/'runs/m4/graph.json')['path'])
        annotation_path=root/plan['annotations']
        if sha256(annotation_path)!=plan['annotations_sha256']:
            raise ValueError('Annotation export changed')
        annotation=json.loads(gzip.decompress(annotation_path.read_bytes()))
        if (annotation['source_sha256']!=sha256(graph_path/'annotations.feather')
                or annotation['graph_id']!=read(graph_path/'manifest.json')['graph_id']):
            raise ValueError('Annotation export differs from this graph')
        classes=np.asarray([x or 'unannotated' for x in annotation['superclass']])
        source=dict(plan_sha256=sha256(args.plan),worker_sha256=sha256(Path(__file__)),
                    helper_sha256=sha256(Path(flow.__file__)),runtime_sha256=runtime_hashes(),
                    scipy=scipy.__version__,numpy=np.__version__,bank_sha256=sha256(bank_dir/'bank.json'),
                    selection_sha256=sha256(out/'selection.npz'),annotation_sha256=sha256(graph_path/'annotations.feather'))
        atomic_json(out/'source.json',source)
        with np.load(root/plan['input_map'],allow_pickle=False) as data:
            visual=data['sensors'];context=data['context_nodes']
        records=[]
        for checkpoint_step in plan['checkpoint_steps']:
            directory=out/f'step-{checkpoint_step:08d}';directory.mkdir()
            checkpoint=root/'runs'/case['run_id']/'checkpoints'/f'step-{checkpoint_step:08d}.npz'
            receipt=read(checkpoint.with_suffix('.json'))
            if receipt['replica_status']!='verified' or sha256(checkpoint)!=receipt['sha256']:
                raise ValueError('Checkpoint needs a verified replica')
            player,metadata=load_player(checkpoint,graph_path,threads=len(plan['cpus']))
            if (player.config.steps!=plan['passes'] or player.config.rate_softness or player.config.readout_mean_scale!=1
                    or player.config.seed!=args.seed or player.mode!=plan['input_mode']
                    or metadata['dataset_id']!=manifest['dataset_id']
                    or metadata['input_contract']['attachment_sha256']!=sha256(root/plan['input_map'])):
                raise ValueError('Unsupported source model')
            graph=player.graph;params=player.parameters();before={k:array_hash(v) for k,v in params.items()}
            ports=player.ports;n=len(graph['type_id']);motors=expected_motors
            np.testing.assert_array_equal(ports['output_group'][motors],np.arange(len(motors)))
            np.testing.assert_array_equal(ports['output_scale'][motors],1)
            groups=dict(all=np.arange(n),visual_sensors=visual,context_sensors=context,motors=motors,
                        other=np.setdiff1d(np.arange(n),np.r_[visual,context,motors]))
            groups.update({'superclass/'+name:np.flatnonzero(classes==name) for name in sorted(set(classes))})
            linear=flow.Linearization(graph,params)
            if checkpoint_step==0:
                eye_distance=flow.distances(graph['indptr'],graph['src'],visual,plan['passes']-1)
                motor_distance=flow.distances(graph['indptr'],graph['src'],motors,plan['passes']-1,reverse=True)
                visual_edges=(eye_distance[graph['src']]>=0)&(motor_distance[graph['dst']]>=0)&(
                    eye_distance[graph['src']]+1+motor_distance[graph['dst']]<=plan['passes']-1)
                np.savez(out/'distances.npz',from_eye=eye_distance,to_motor=motor_distance,visual_edges=visual_edges)
            signals={};retained=None;reconstruction=[];bounds=[]
            for begin in range(0,count,batch):
                atomic_json(out/'status.json',dict(state='forward',checkpoint_step=checkpoint_step,positions=begin,updated=time.time()))
                encoded=player.adapter.encode(features[begin:begin+batch],plan['input_mode'])
                current=player.core.infer(encoded,trace=True)
                neutral=player.core.infer(player.adapter.encode(features[begin:begin+batch],'neutral'),trace=True)
                for step,(actual,other) in enumerate(zip(current['states'],neutral['states'])):
                    signals.setdefault(step,[]).append((actual.copy(),other.copy()))
                if begin==0:retained=(encoded,current)
                if checkpoint_step==1000:
                    bank_motors=np.load(bank_dir/'train-motors.npy',mmap_mode='r',allow_pickle=False)
                    np.testing.assert_array_equal(np.maximum(current['states'][-1][motors].T,0),bank_motors[begin:begin+batch])
                    np.testing.assert_array_equal(current['logits'],labels['baseline'][begin:begin+batch])
            per_node={};signal_records=[]
            for step,blocks in signals.items():
                current=np.concatenate([b[0] for b in blocks],axis=1)
                neutral=np.concatenate([b[1] for b in blocks],axis=1)
                statistics=flow.node_signal(current,neutral)
                per_node.update({f'{step}/{key}':value for key,value in statistics.items()})
                signal_records.append(dict(step=step,groups={name:{key:flow.summary(value[nodes]) for key,value in statistics.items()} for name,nodes in groups.items()}))
            del signals,current,neutral
            np.savez(directory/'node-signals.npz',**per_node);del per_node
            encoded,output=retained;states=output['states']
            drive=np.zeros((n,batch),np.float32);sensory=np.flatnonzero(ports['input_index']>=0);inputs=ports['input_index'][sensory]
            drive[sensory]=params['input_gain'][inputs,None]*encoded.T[inputs]
            for step in range(plan['passes']):
                previous=states[step].astype(float)
                predicted=(1-linear.alpha[:,None])*previous+linear.alpha[:,None]*(linear.matrix@np.maximum(previous,0)+params['bias'][graph['type_id'],None]+drive)
                reconstruction.append(compare(predicted,states[step+1],dict(rtol=3e-4,atol=3e-6)))
                bound=linear.local_bound(states[step])
                bounds.append(dict(step=step,groups={name:flow.summary(bound[nodes]) for name,nodes in groups.items()}))
            l1=np.bincount(graph['dst'],weights=np.abs(linear.weights),minlength=n)
            l2=np.sqrt(np.bincount(graph['dst'],weights=linear.weights**2,minlength=n))
            fanin=np.divide(l1*l1,l2*l2,out=np.zeros(n),where=l2>0)
            weight_summary={name:{key:flow.summary(value[nodes]) for key,value in dict(l1=l1,l2=l2,effective_fanin=fanin).items()} for name,nodes in groups.items()}
            residual=dict(legal=labels['legal'][:batch])
            if checkpoint_step==1000:
                fit=root/'runs/motor-convergence-v1'/f'seed-{args.seed}'/'scaled-linear/ridge-0.01'
                fit_report=read(fit/'result.json')
                fit_source=read(fit.parent.parent/'source.json')
                if (sha256(fit/'checkpoint.npz')!=fit_report['checkpoint_sha256']
                        or sha256(fit.parent/'basis.npz')!=fit_report['basis_sha256']
                        or fit_source['bank_sha256']!=case['bank_sha256'] or not fit_report['sufficiently_converged']):
                    raise ValueError('Fitted decoder bytes changed')
                with np.load(fit/'checkpoint.npz',allow_pickle=False) as data:residual.update(weight=data['weight'],bias=data['bias'])
                with np.load(fit.parent/'basis.npz',allow_pickle=False) as data:residual['center']=data['mean']
                with np.load(bank_dir/'normalization.npz',allow_pickle=False) as data:residual.update(mean=data['mean'],scale=data['scale'])
            gradients=[]
            for name in plan['gradient_cases']:
                if name=='fitted-linear-policy' and checkpoint_step==0:continue
                atomic_json(out/'status.json',dict(state='adjoint',checkpoint_step=checkpoint_step,objective=name,updated=time.time()))
                extra,score,pooled,loss=flow.pooled_cotangent(name,output,params,motors,labels['policy'][:batch],labels['target_value'][:batch],residual)
                _,native=player.core.embedding_loss_and_grad(encoded,lambda embedding,value:(loss,extra,score))
                last=np.zeros((n,batch),np.float32)
                last[motors]=((pooled.T*ports['output_scale'][motors,None])*params['readout_gain'][motors,None])*(states[-1][motors]>0)
                grad,ddrive,cotangents=linear.adjoint(states,last)
                grad['input_gain']=flow.input_gradient(ddrive,encoded,ports,player.config.features)
                readout=np.zeros(n)
                readout[motors]=(pooled.T.astype(float)*np.maximum(states[-1][motors],0)*ports['output_scale'][motors,None]).sum(axis=1)
                grad['readout_gain']=readout
                errors={key:compare(value,native[key],plan['native_comparison']) for key,value in grad.items()}
                node_arrays={f'{step}/rms':np.sqrt(np.mean(g*g,axis=1)) for step,g in enumerate(cotangents)}
                node_arrays.update({f'{step}/l1':np.sum(np.abs(g),axis=1) for step,g in enumerate(cotangents)})
                np.savez(directory/(name+'-node-gradients.npz'),**node_arrays,drive_rms=np.sqrt(np.mean(ddrive*ddrive,axis=1)))
                record=dict(objective=name,loss=loss,comparisons=errors,
                    native_gradients={key:flow.gradient_summary(value,plan['epsilon']) for key,value in native.items()},
                    edge_path_gradients={label:flow.gradient_summary(native['edge'][mask],plan['epsilon']) for label,mask in [('within_visual_paths',visual_edges),('outside_visual_horizon',~visual_edges)]},
                    cotangents=[dict(step=step,groups={label:dict(rms=flow.summary(node_arrays[f'{step}/rms'][nodes]),l1=float(node_arrays[f'{step}/l1'][nodes].sum())) for label,nodes in groups.items()}) for step in range(len(states))],
                    input_drive_groups={label:flow.gradient_summary(ddrive[nodes]) for label,nodes in groups.items()},
                    native_gradient_sha256={key:array_hash(value) for key,value in native.items()})
                atomic_json(directory/(name+'.json'),record);gradients.append(record)
                if not all(row['passed'] for row in errors.values()):
                    np.savez(directory/(name+'-failed-gradients.npz'),**{'native/'+k:native[k] for k in grad},**{'reference/'+k:v for k,v in grad.items()})
                    raise ValueError('Independent adjoint differs from native gradients; preserve this failed audit')
                del native,grad,ddrive,cotangents,node_arrays
            moments={}
            with np.load(checkpoint,allow_pickle=False) as saved:
                for key,value in params.items():
                    second=saved['second/'+key].astype(float)
                    rms=np.sqrt(second/(1-.999**checkpoint_step)) if checkpoint_step else np.zeros_like(second)
                    moments[key]=dict(second_moment_rms=flow.summary(rms),epsilon_visibility=flow.summary(rms/(rms+plan['epsilon'])))
            changes={}
            initial=root/'runs'/case['run_id']/'checkpoints/step-00000000.npz'
            with np.load(initial,allow_pickle=False) as saved:
                for key,value in params.items():
                    original=saved['param/'+key]
                    changes[key]=dict(changed=int(np.count_nonzero(original!=value)),parameters=len(value),delta=flow.summary(value.astype(float)-original))
            if before!={key:array_hash(value) for key,value in player.parameters().items()}:
                raise ValueError('Read-only audit changed parameters')
            record=dict(step=checkpoint_step,checkpoint_sha256=receipt['sha256'],forward_reconstruction=reconstruction,
                signals=signal_records,weights=weight_summary,jacobian_row_bound=bounds,gradients=gradients,
                adam_moments=moments,parameter_changes=changes,parameters_unchanged=True)
            atomic_json(directory/'result.json',record);records.append(record)
            if not all(r['passed'] for r in reconstruction):raise ValueError('Recorded recurrence does not reconstruct')
            del player,linear,states,retained,output
        result=dict(status='complete',seed=args.seed,source=source,records=records,seconds=time.time()-started,
                    visual_path_edges=int(visual_edges.sum()),motor_eye_distance=flow.summary(eye_distance[motors]),
                    scope=plan['scope'],quantiles=flow.QUANTILES)
        atomic_json(out/'result.json',result);atomic_json(out/'status.json',dict(state='complete',updated=time.time()))
        print(json.dumps(dict(status='complete',seed=args.seed,seconds=result['seconds'],visual_path_edges=result['visual_path_edges'])),flush=True)


if __name__=='__main__':main()
