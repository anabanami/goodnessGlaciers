"""Run weighted_anisotropy.py on the DEM_slicer window CSVs.

weighted_anisotropy.py imports Tee, PROCESSING_FLAG_NOTE, processing_flag_of, flag_suptitle
and OUTPUT_BASE_PATH from the ODSA pipeline. This module executes its source with those
imports removed and the five names supplied directly, then calls the fits.

    python run_wa.py                                    # every CSV in window_csvs/
    python run_wa.py window_csvs/Dubawnt_window_stats.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

HERE = Path(__file__).resolve().parent
SRC = HERE / 'weighted_anisotropy.py'
OUT = HERE / 'anisotropy_out'

DROP = ('from config', 'from plotting', 'from loading', 'OUTPUT_BASE_PATH =')


def load():
    src = [l for l in SRC.read_text().splitlines(keepends=True) if not l.startswith(DROP)]
    wa = {'__name__': 'weighted_anisotropy'}
    exec(compile(''.join(src), str(SRC), 'exec'), wa)
    wa |= {'OUTPUT_BASE_PATH': str(OUT) + '/',
           'PROCESSING_FLAG_NOTE': {},
           'processing_flag_of': lambda df: None,
           'flag_suptitle': lambda fig, title, flag, fontsize=14:
               fig.suptitle(title, fontsize=fontsize)}
    return wa


if __name__ == '__main__':
    csvs = [Path(a) for a in sys.argv[1:]] or sorted((HERE / 'window_csvs')
                                                     .glob('*_window_stats.csv'))
    if not csvs:
        raise SystemExit('no window CSVs — run DEM_slicer.py first')

    wa = load()
    OUT.mkdir(exist_ok=True)
    for csv in csvs:
        print(f'\n{"=" * 70}\n{csv.name}\n{"=" * 70}')
        wa['plot_anisotropy'](str(csv), level='window')
        wa['local_anisotropy'](str(csv))
    print(f'\n-> {OUT}')
