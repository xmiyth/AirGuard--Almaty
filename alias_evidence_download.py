"""Snapshot public upstream documentation only; never download replacement readings."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import requests

OUT = Path(__file__).resolve().parent / 'experiments' / 'station_alias_audit_20261114' / 'evidence'
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / 'download_manifest.json').exists():
    raise SystemExit('Evidence already snapshotted; refusing overwrite')
base = 'https://api.github.com/repos/qazybekb/AirDatakz-OpenData'
response = requests.get(base + '/commits/main',timeout=30)
response.raise_for_status()
commit = response.json()['sha']
(OUT / 'commit.json').write_bytes(response.content)
records = []
for name in ['README.md','DATASHEET.md','DATA_DICTIONARY.md','stations.csv','station_periods.csv']:
    url = f'https://raw.githubusercontent.com/qazybekb/AirDatakz-OpenData/{commit}/{name}'
    response = requests.get(url,timeout=30)
    response.raise_for_status()
    (OUT / name).write_bytes(response.content)
    records.append(dict(file=name,url=url,sha256=hashlib.sha256(response.content).hexdigest(),
                        retrieved_at_utc=datetime.now(timezone.utc).isoformat()))
(OUT / 'download_manifest.json').write_text(json.dumps(dict(commit=commit,files=records),indent=2))
print(json.dumps(dict(commit=commit,files=[r['file'] for r in records]),indent=2))
