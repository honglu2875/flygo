#!/usr/bin/env python3
"""Freeze a full-size CNN CPU reference before TPU qualification or learning."""
import argparse
import json
import os
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--cpus',default='117,118,119')
    p.add_argument('--channels',type=int,default=64)
    p.add_argument('--blocks',type=int,default=10)
    args=p.parse_args();cpus=list(map(int,args.cpus.split(',')))
    from flygo.runtime import pin
    from flygo.storage import StorageBudget,GIB
    pin(cpus)
    with StorageBudget(args.root).reserve(files=128*(1<<20),heap=4*GIB,purpose='CNN CPU numerical reference'):
        from flygo.jax.control import prepare
        report=prepare(args.root,args.output,threads=len(cpus),channels=args.channels,blocks=args.blocks)
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
