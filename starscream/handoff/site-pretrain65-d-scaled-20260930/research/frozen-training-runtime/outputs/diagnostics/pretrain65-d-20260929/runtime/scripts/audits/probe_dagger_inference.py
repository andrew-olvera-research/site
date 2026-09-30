"""Benchmark tiny DAgger inference transactions and action parity on real replay."""
import argparse,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch,h5py
from scripts import train_privileged_racing as t
from starscream.inference_graph import HostPolicyInferenceGraphs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes',default='eager,compiled,eager_strict,compiled_strict')
    parser.add_argument('--output',default='outputs/dagger-throughput/inference.json')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    import torch._inductor.config as config
    config.compile_threads=1
    root=Path('outputs/checkpoints/starscream-v6.21.1.update-fix-pretrain65-dagger')
    policy,_,_,_=t.load_policy_checkpoint(root/'latest.pt','cuda');policy.eval()
    with h5py.File(root/'dagger-replay/round-00218.h5') as f:
        h=f['online/histories'][:1024].astype(np.float32)
        s=f['online/speed_commands'][:1024]
    original=policy.forward
    results={}; references={}
    for mode in args.modes.split(','):
        torch.set_float32_matmul_precision('highest' if 'strict' in mode else 'high')
        options={'triton.cudagraphs':False}
        if mode=='compiled_precise':options.update(force_same_precision=True,emulate_precision_casts=True)
        policy.forward=original if mode.startswith('eager') else torch.compile(original,fullgraph=True,dynamic=True,
            options=options)
        graph=HostPolicyInferenceGraphs(policy,16)
        rows=[]
        for size in [1,2,4,8,16]:
            output=graph.predict_numpy(h[:size],s[:size])
            start=time.perf_counter()
            for _ in range(1000): graph.predict_numpy(h[:size],s[:size])
            seconds=(time.perf_counter()-start)/1000
            # Check many actual states, not only the timed first microbatch.
            outputs=np.concatenate([graph.predict_numpy(a,b) for a,b in zip(np.split(h,len(h)//size),np.split(s,len(s)//size))])
            if mode=='eager': references[size]=outputs
            rows.append(dict(batch=size,seconds=seconds,max_action_error=float(np.max(np.abs(outputs-references[size])))))
            print(mode,rows[-1],flush=True)
        results[mode]=rows
    Path(args.output).write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
