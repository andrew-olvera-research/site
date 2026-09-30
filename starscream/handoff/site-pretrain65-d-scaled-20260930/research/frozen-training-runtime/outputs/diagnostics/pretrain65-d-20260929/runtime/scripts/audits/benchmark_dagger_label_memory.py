"""Reproduce packed-response label retention with real replay values, without solving."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
from starscream.dagger_row_buffer import NumericRowBuffer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['list','chunks'],required=True)
    parser.add_argument('--rows',type=int,default=200000)
    args=parser.parse_args()
    with h5py.File('outputs/checkpoints/starscream-v6.21.1.update-fix-pretrain65-dagger/dagger-replay/round-00218.h5') as archive:
        source={k:archive['online'][k][:1024] for k in ['histories','actions','previous','dynamics']}
    result={k:NumericRowBuffer() if args.mode=='chunks' else [] for k in source}
    start=time.perf_counter()
    for first in range(0,args.rows,1024):
        episode={k:[] for k in source}
        for index in range(first,min(first+1024,args.rows)):
            row=index%1024
            # The current packed response has 142 float32 values. Retained
            # labels reference its first 27; features/diagnostics are transient.
            packet=np.zeros(142,np.float32)
            packet[:4]=source['actions'][row]
            packet[4:8]=source['previous'][row]
            packet[8:27]=source['dynamics'][row]
            episode['histories'].append(source['histories'][row].copy())
            for key,a,b in [('actions',0,4),('previous',4,8),('dynamics',8,27)]:
                episode[key].append(packet[a:b])
        for key in result:
            result[key].extend(episode[key])
    elapsed=time.perf_counter()-start
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
    digest=hashlib.sha256()
    for key,rows in result.items():
        digest.update(np.asarray(rows).tobytes())
    record=dict(mode=args.mode,rows=args.rows,seconds=elapsed,retention_peak_rss_gib=peak,digest=digest.hexdigest(),
                note='Real replay values repeated into simulated response buffers; numeric label storage only, not a full collector RSS measurement.')
    Path(f'outputs/dagger-throughput/label-memory-{args.mode}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(record)


if __name__=='__main__':
    main()
