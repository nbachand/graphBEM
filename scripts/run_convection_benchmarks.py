"""Run the convection comparison matrix against existing EnergyPlus cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Skip completed runs")
    args = parser.parse_args()
    runs = []
    for climate, pattern in [('burbank', '*BURBANK*'), ('palm_springs', '*PALM-SPRINGS*'), ('arcata', '*ARCATA*')]:
        case = next(args.case_root.glob(pattern))
        epw = ROOT/'energyPlus/weather/CAClimateZones'/case.name/(case.name+'.epw')
        variants = [('variable', [])] + [(f'fixed_{h}', ['--interior', 'fixed', '--exterior',
                     'doe2_fixed_natural', '--h-natural', str(h)]) for h in [1, 2, 3]]
        if climate == 'burbank':
            variants += [('refined', ['--dt', '30', '--cells', '36']),
                         ('isotropic', ['--sky', 'isotropic'])]
        else:
            variants += [('legacy', ['--interior', 'legacy', '--exterior', 'legacy', '--sky', 'isotropic'])]
        for variant, options in variants:
            output = ROOT/'analysis/convection_models'/f'{climate}_{variant}'
            runs.append((output, [sys.executable, str(ROOT/'scripts/compare_ep_free_running.py'),
                '--ep-case', str(case), '--epw', str(epw), '--output', str(output), *options]))
    def run(item):
        output, command = item
        if args.resume and (output/"metadata.json").exists():
            print("Already complete", output.name, flush=True)
            return
        output.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1')
        print('Starting', output.name, flush=True)
        with (output/'run.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('Finished', output.name, flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run, runs))


if __name__ == '__main__':
    main()
