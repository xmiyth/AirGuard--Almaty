"""Archive public provider pages and availability failures, without credentials."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import requests
import hashlib
import json

OUT=Path(__file__).resolve().parent/'experiments/station_alias_audit_20261114/evidence/provider'
OUT.mkdir(exist_ok=True)
if (OUT/'manifest.json').exists():
    raise SystemExit('Refusing overwrite')
urls={
    'waqi_clarity_arena.html':'https://aqicn.org/station/kazakhstan-alatau-district-almaty-arena/',
    'waqi_clarity_road.html':'https://aqicn.org/station/clarity/dxkuv4229/',
    'waqi_embassy.html':'https://aqicn.org/data-platform/api/H12774/',
    'openaq_arena.html':'https://explore.openaq.org/locations/2812550',
    'openaq_road.html':'https://explore.openaq.org/locations/2812831',
    'openaq_api_arena.json':'https://api.openaq.org/v3/locations/2812550',
}
def fetch(item):
    name,url=item
    row=dict(file=name,url=url,retrieved_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        r=requests.get(url,timeout=20)
        (OUT/name).write_bytes(r.content)
        row.update(status=r.status_code,final_url=r.url,sha256=hashlib.sha256(r.content).hexdigest(),bytes=len(r.content))
    except requests.RequestException as exc:
        row['error']=str(exc)
    return row
with ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(fetch,urls.items()))
(OUT/'manifest.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
