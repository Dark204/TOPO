"""Download the fixed CARE source files and verify their SHA-256 hashes."""
from pathlib import Path
import hashlib
import json
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
REPOSITORY='M-panahandeh/CARE-Context-Aware-Root-Cause-Identification-Using-Distributed-Traces-and-Profiling-Metrics'

def main():
    config=json.loads((ROOT/'parameters.json').read_text())
    for name,expected in config['author_files_sha256'].items():
        destination=ROOT/'code/upstream'/name
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()==expected:
            continue
        url=f"https://raw.githubusercontent.com/{REPOSITORY}/{config['author_commit']}/{name}"
        with urlopen(url,timeout=60) as response: content=response.read()
        if hashlib.sha256(content).hexdigest()!=expected:
            raise ValueError(f'Upstream source hash mismatch: {name}')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(content)
    print('CARE source files verified.')

if __name__=='__main__':main()
