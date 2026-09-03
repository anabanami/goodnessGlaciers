"""Null floor over seeds. Generate isotropic beds (DEMs).

Each DEM is 293 MB, written to null_seeds/seed_NNN/.

    python null_seeds.py                    # seeds 1-20
    python null_seeds.py --n 5              # seeds 1-5
    python null_seeds.py --n 5 --start 100  # seeds 100-105
    python null_seeds.py -- --beta-1d 1.73  # everything after -- goes to the synthetic_isotropic.py generator

"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

FACTORY = Path.home() / 'Desktop/code/bedrock_factory/synthetic_isotropy'
GEN = FACTORY / 'synthetic_isotropic.py'
WORK = HERE / 'null_seeds'
SLOPE_REF_DEM = FACTORY / 'Site F Nunavut/prep/Site F Nunavut_dem.tif'
SLOPE_REF_WATER = FACTORY / 'Site F Nunavut/prep/Site F Nunavut_water.tif'


def seed_name(seed):
    return f'seed_{seed:03d}'


def generate(seed, name, passthru=()):
    # The generator writes relative filenames, so we run it in the seed folder
    out = WORK / name
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(GEN), '--seed', str(seed), '--name', name,
                    '--slope-ref-dem', str(SLOPE_REF_DEM),
                    '--slope-ref-water', str(SLOPE_REF_WATER), *passthru],
                   cwd=out, check=True)
    return out


def parse_args(argv):
    if '--' in argv:
        cut = argv.index('--')
        argv, passthru = argv[:cut], argv[cut + 1:]
    else:
        passthru = []
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--start', type=int, default=1)
    p.add_argument('--force', action='store_true', help='regenerate seeds that already have a DEM')
    return p.parse_args(argv), passthru


def main():
    args, passthru = parse_args(sys.argv[1:])
    seeds = range(args.start, args.start + args.n)
    for i, seed in enumerate(seeds, 1):
        name = seed_name(seed)
        if not args.force and (WORK / name / f'{name}_dem.tif').exists():
            print(f'[{i}/{len(seeds)}] {name}: DEM exists, skipping')
            continue
        print(f'[{i}/{len(seeds)}] {name}: generating')
        generate(seed, name, passthru)


if __name__ == '__main__':
    main()
