"""Read current Linux/WSL and container limits without allocating memory."""
import os
from pathlib import Path


def memory_pressure_snapshot(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    result = {}
    try:
        values = {line.split(':')[0]: int(line.split()[1])*1024
                  for line in (proc/'meminfo').read_text().splitlines() if ':' in line}
        result.update(memory_total_gib=values['MemTotal']/2**30,
                      memory_available_gib=values['MemAvailable']/2**30,
                      swap_used_gib=(values.get('SwapTotal',0)-values.get('SwapFree',0))/2**30)
        result['memory_effective_total_gib'] = result['memory_total_gib']
        result['memory_effective_available_gib'] = result['memory_available_gib']
    except (OSError, KeyError, ValueError):
        pass
    try:
        current = int((cgroup/'memory.current').read_text())/2**30
        result['cgroup_memory_gib'] = current
        raw_limit = (cgroup/'memory.max').read_text().strip()
        if raw_limit != 'max':
            limit = int(raw_limit)/2**30
            result['cgroup_memory_limit_gib'] = limit
            result['memory_effective_total_gib'] = min(result.get('memory_total_gib',limit),limit)
            result['memory_effective_available_gib'] = min(
                result.get('memory_available_gib',limit),max(0.,limit-current))
    except (OSError, ValueError):
        pass
    try:
        vm = dict(line.split() for line in (proc/'vmstat').read_text().splitlines())
        page_size = os.sysconf('SC_PAGE_SIZE')
        result['memory_swap_in_bytes_total'] = int(vm['pswpin'])*page_size
        result['memory_swap_out_bytes_total'] = int(vm['pswpout'])*page_size
        result['memory_major_faults_total'] = int(vm['pgmajfault'])
    except (OSError, KeyError, ValueError, AttributeError):
        pass
    try:
        for line in (proc/'pressure/memory').read_text().splitlines():
            parts = line.split()
            fields = dict(part.split('=') for part in parts[1:])
            result[f'memory_psi_{parts[0]}_stall_seconds_total'] = int(fields['total'])/1e6
    except (OSError, KeyError, ValueError):
        pass
    return result


def memory_pressure_delta(before, after, seconds):
    result = {}
    for key in ('memory_swap_in_bytes_total', 'memory_swap_out_bytes_total',
                'memory_major_faults_total', 'memory_psi_some_stall_seconds_total',
                'memory_psi_full_stall_seconds_total'):
        if key in before and key in after:
            change = max(0.,after[key]-before[key])
            result[key.removesuffix('_total')+'_per_second'] = change/max(seconds,1e-6)
    return result
