"""Fetch ArcticDEM tiles, bedform vectors and water polygons for the four validation sites.

Bboxes come from the FRIDGE selection manifests in data/. Mosaic not strips: 2 m strips are
~16 km wide and share the orbit bearing, so seam density would vary with azimuth and forge a
Δβ. Reasoning in TESTING_ANISOTROPY🛠️.md.

    python fetch_dems.py --dry-run           # plan only, downloads nothing
    python fetch_dems.py                     # 10 m mosaic + PANGAEA vectors     ~7.4 GB
    python fetch_dems.py --hydro             # + CanVec 250K water, NU+NT        ~0.6 GB
    python fetch_dems.py --hydro50k          # 50K water instead, small ponds    ~7.0 GB
    python fetch_dems.py --strips            # + 2 m strip selection            ~19.5 GB
    python fetch_dems.py --verify            # HEAD local files, refetch short ones
    python fetch_dems.py --verify --hydro    # flags combine
    python fetch_dems.py --tile 24_18 --site Dubawnt   # one mosaic tile the manifest misses

Resumable: complete files are skipped, partials land in .part and are redone. Never run two
copies at once — they share .part paths. Writes to data/<site>/mosaic_10m/, data/pangaea/,
data/hydro/.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / 'data'
STAC = 'https://stac.pgc.umn.edu/api/v1/search'
COLLECTION = 'arcticdem-mosaics-v4.1-10m'
ASSETS = ['dem', 'mad', 'count', 'maxdate', 'mindate']   # mad/count/dates are the noise-floor terms

PANGAEA = 'https://download.pangaea.de/dataset/939999/files/'
PANGAEA_FILES = [# 'MClintockChannel_Canada.zip',   # site dropped, see archive/README.md
                 'PrinceofWalesIsland_Canada.zip',
                 'Nunavut_Canada.zip', 'McKenzie_Bedform_RawData.xls']

# Water mask. All four sites are in NU; Dubawnt's western edge may cross into NT.
CANVEC = 'https://ftp.maps.canada.ca/pub/nrcan_rncan/vector/canvec/shp/Hydro/'
CANVEC_TERR = ['NU', 'NT']


def coords(arr):
    """Yield (lon, lat) pairs from a nested GeoJSON coordinate array."""
    if arr and all(isinstance(v, (int, float)) for v in arr):
        yield arr[0], arr[1]
        return
    for sub in arr:
        if isinstance(sub, list):
            yield from coords(sub)


def bbox(site):
    feats = json.loads((site / 'fridge_export.json').read_text())['features']
    pts = [p for f in feats for p in coords(f['geometry']['coordinates'])]
    lon, lat = zip(*pts)
    return [min(lon), min(lat), max(lon), max(lat)]


def tile_bbox(name, edge=20):
    """Lon/lat bbox of a named ArcticDEM mosaic tile, from its EPSG:3413 footprint.

    Tile ROW_COL covers 100 km from ((COL - 41), (ROW - 41)) * 100 km. The boundary is
    sampled rather than cornered, since a tile's edges are curved in lon/lat.
    """
    from pyproj import Transformer
    row, col = (int(v) for v in name.split('_'))
    left, bottom = (col - 41) * 1e5, (row - 41) * 1e5
    xs, ys = [], []
    for i in range(edge + 1):
        f = i / edge
        xs += [left + f * 1e5, left + f * 1e5, left, left + 1e5]
        ys += [bottom, bottom + 1e5, bottom + f * 1e5, bottom + f * 1e5]
    lon, lat = Transformer.from_crs('EPSG:3413', 'EPSG:4326', always_xy=True).transform(xs, ys)
    return [min(lon), min(lat), max(lon), max(lat)]


def stac_items(bb):
    body = json.dumps({'collections': [COLLECTION], 'bbox': bb, 'limit': 500}).encode()
    req = urllib.request.Request(STAC, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)['features']


PLANNED = []
UA = {'User-Agent': 'ODSA-anisotropy-validation/1.0 (research; ana.fabelah@gmail.com)'}
RETRIES = 5   # PANGAEA throttles with 503s; PGC occasionally drops a connection


def remote_size(url):
    try:
        req = urllib.request.Request(url, headers=UA, method='HEAD')
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get('Content-Length') or 0)
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0


def get(url, dest, dry=False, base_wait=5, verify=False):
    if dest.exists() and dest.stat().st_size > 0:
        if not verify:
            print(f'    have  {dest.name}')
            return 0
        # Files pulled before the truncation guard existed were never length-checked.
        want, have = remote_size(url), dest.stat().st_size
        if want and want != have:
            print(f'    SHORT {dest.name}: {have} of {want} bytes — refetching')
            dest.unlink()
        else:
            print(f'    ok    {dest.name}')
            return 0
    if dry:
        PLANNED.append(dest.name)
        print(f'    would {dest.name}')
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, 'wb') as f:
                expect = int(r.headers.get('Content-Length') or 0)
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            # A dropped connection can end the loop without raising; a short file must not
            # be renamed, or it would look complete and be skipped on every later run.
            if expect and tmp.stat().st_size != expect:
                raise OSError(f'truncated: {tmp.stat().st_size} of {expect} bytes')
            break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            tmp.unlink(missing_ok=True)
            if attempt == RETRIES - 1:
                print(f'    FAIL  {dest.name}: {e}')
                return 0
            wait = base_wait * 2 ** attempt
            print(f'    retry {dest.name} in {wait}s ({e})')
            time.sleep(wait)
    tmp.rename(dest)
    mb = dest.stat().st_size / 1e6
    print(f'    got   {dest.name}  {mb:.1f} MB')
    return mb


if __name__ == '__main__':
    args = sys.argv[1:]
    dry, strips = '--dry-run' in args, '--strips' in args
    hydro = '--hydro' in args or '--hydro50k' in args
    verify = '--verify' in args
    if '--tile' in args:
        name = args[args.index('--tile') + 1]
        site = DATA / args[args.index('--site') + 1]
        items = [it for it in stac_items(tile_bbox(name)) if it['id'].startswith(name + '_')]
        print(f'{name}: {len(items)} matching tiles into {site}/mosaic_10m')
        got = 0.0
        for it in items:
            for k in [k for k in ASSETS if k in it['assets']]:
                href = it['assets'][k]['href']
                got += get(href, site / 'mosaic_10m' / Path(href).name, dry, verify=verify)
        print(f'fetched {got / 1000:.2f} GB')
        sys.exit()

    sites = sorted(p for p in DATA.iterdir() if (p / 'fridge_export.json').exists())
    if not sites:
        sys.exit(f'no fridge_export.json under {DATA}')

    total = 0.0
    for site in sites:
        bb = bbox(site)
        print(f"\n{site.name}  bbox {['%.3f' % v for v in bb]}")

        try:
            items = stac_items(bb)
        except (urllib.error.URLError, TimeoutError, KeyError) as e:
            print(f'  STAC query failed ({e}) — fetch these tiles by hand for the bbox above')
            continue

        print(f'  {len(items)} mosaic tiles')
        for it in items:
            keys = [k for k in ASSETS if k in it['assets']]
            if not keys:
                print(f"  {it['id']}: no expected assets, has {sorted(it['assets'])}")
                continue
            for k in keys:
                href = it['assets'][k]['href']
                total += get(href, site / 'mosaic_10m' / Path(href).name, dry, verify=verify)

        if strips:
            feats = json.loads((site / 'fridge_export.json').read_text())['features']
            print(f'  {len(feats)} 2 m strips')
            for f in feats:
                url = f['properties']['pgc_download_link']
                total += get(url, site / 'strips_2m' / Path(url).name, dry, verify=verify)

    # PANGAEA rate-limits per IP: slow base backoff, and space the requests out.
    print('\nPANGAEA bedform vectors')
    for i, name in enumerate(PANGAEA_FILES):
        if i and not dry:
            time.sleep(15)
        total += get(PANGAEA + name, DATA / 'pangaea' / name, dry, base_wait=30, verify=verify)

    if hydro:
        scale = '50K' if '--hydro50k' in args else '250K'
        print(f'\nCanVec {scale} hydro')
        for terr in CANVEC_TERR:
            name = f'canvec_{scale}_{terr}_Hydro_shp.zip'
            total += get(CANVEC + name, DATA / 'hydro' / name, dry, verify=verify)

    if dry:
        gb = sum(f['properties']['dem_filesize'] for s in sites
                 for f in json.loads((s / 'fridge_export.json').read_text())['features']) if strips else 0
        print(f'\nwould fetch {len(PLANNED)} files'
              + (f', of which {gb:.1f} GB of 2 m strips' if strips else ''))
    else:
        print(f'\nfetched {total / 1000:.2f} GB')
    if not strips:
        print('2 m strips skipped — rerun with --strips for the co-registration cross-check')
