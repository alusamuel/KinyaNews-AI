"""Build and smoke-test a minimal deployment ZIP without training data."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def build():
    output = ROOT / 'dist'
    output.mkdir(exist_ok=True)
    files = ['.python-version', 'app.py', 'requirements.txt', 'vercel.json', 'src/__init__.py',
             'src/labels.py', 'models/classifier.joblib', 'models/metadata.json']
    files += [str(p.relative_to(ROOT)) for folder in ('templates', 'static')
              for p in (ROOT / folder).rglob('*') if p.is_file()]
    for name in files:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
    archive = output / 'kinyanews-deployment.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            z.write(ROOT / name, name)
    # New interpreter and extracted package: no access to project data or candidates.
    with tempfile.TemporaryDirectory(prefix='kinyanews-package-') as folder:
        with zipfile.ZipFile(archive) as z:
            z.extractall(folder)
        subprocess.run([sys.executable, '-c', '''
from app import app
c=app.test_client()
assert c.get('/').status_code == 200
assert c.get('/health').json['model_loaded'] is True
for route in ['/static/app.js','/static/style.css']:
    assert c.get(route).status_code == 200
r=c.post('/api/predict',json={'text':'Ikipe yatsinze umukino w umupira wabereye mu mujyi wa Kigali.'})
assert r.status_code == 200 and r.json['label']
assert c.post('/api/predict',json={'text':'short'}).status_code == 400
print('Extracted deployment package: homepage, assets, health, inference and validation passed.')
'''], cwd=folder, check=True)
    manifest = {'archive': archive.name, 'bytes': archive.stat().st_size,
                'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'files': sorted(files), 'isolated_package_smoke_test': 'passed',
                'public_deployment_verified': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    build()
