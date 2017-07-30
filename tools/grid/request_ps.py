import json
import os

import requests

with open(f"{os.environ['CLUSTER_SPEC']}.json", 'r') as jf:
    cluster_spec = json.load(jf)

for s in cluster_spec['ps']:
    response = requests.get(f'http://{s}')
    print(f'{s}: {response.status_code}')
