#!/usr/bin/env python3
"""Read CPU/RAM capacity on the four hosts without initializing TPU runtimes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import getpass
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from flygo.storage import StorageBudget

PROBE = r'''
from pathlib import Path
import json, os, socket, time
cpus = sorted(os.sched_getaffinity(0))
cores = set()
for cpu in cpus:
    topology = Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
    cores.add((int((topology/'physical_package_id').read_text()), int((topology/'core_id').read_text())))
memory = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
          if line.startswith(('MemTotal:', 'MemAvailable:', 'Shmem:'))}
filesystem = os.statvfs('/dev/shm')
print(json.dumps(dict(host=socket.gethostname(), measured_unix=time.time(), logical_cpus=len(cpus),
                     physical_cores=len(cores), numa_nodes=len(list(Path('/sys/devices/system/node').glob('node[0-9]*'))),
                     cpu_ids=cpus, memory=memory, load_average=os.getloadavg(),
                     shm_total=filesystem.f_blocks*filesystem.f_frsize,
                     shm_free=filesystem.f_bavail*filesystem.f_frsize)))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", default=[f"t1v-n-a09f5679-w-{i}" for i in range(4)])
    parser.add_argument("--output", type=Path, default=Path("/dev/shm/flygo/runs/m2/inventory.json"))
    args = parser.parse_args()

    def probe(host):
        if not host or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-." for c in host) or host[0] == "-":
            raise ValueError("Invalid host name")
        result = subprocess.run(["ssh", "-F", "/dev/null", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                 getpass.getuser() + "@" + host, "python3", "-"],
                                input=PROBE, text=True, capture_output=True, check=True, timeout=30)
        return json.loads(result.stdout)

    with StorageBudget().reserve(files=1 << 20, heap=16 << 20, purpose="four-host inventory"):
        with ThreadPoolExecutor(max_workers=min(4, len(args.hosts))) as executor:
            hosts = list(executor.map(probe, args.hosts))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(schema_version=1, hosts=hosts), indent=2) + "\n")
    for host in hosts:
        print(json.dumps({key: host[key] for key in ("host", "logical_cpus", "physical_cores", "numa_nodes",
                                                    "memory", "shm_total", "shm_free", "load_average")}))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
