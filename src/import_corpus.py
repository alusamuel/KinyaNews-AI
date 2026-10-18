"""Convert the KINNEWS archive referenced by Hugging Face to local named labels."""
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
from src.labels import LABELS


def main():
    archive = Path('data/KINNEWS.zip')
    frames = []
    with zipfile.ZipFile(archive) as source:
        for split in ('train', 'test'):
            raw = pd.read_csv(source.open(f'KINNEWS/raw/{split}.csv'))
            expected = raw.label.map(lambda value: LABELS[int(value) - 1])
            # The original corpus calls the first class "politic".
            assert raw.en_label.replace({'politic': 'politics'}).eq(expected).all()
            frame = pd.read_csv(source.open(f'KINNEWS/cleaned/{split}.csv'))
            assert frame.label.between(1, len(LABELS)).all()
            frame['label'] = frame.label.map(lambda value: LABELS[int(value) - 1])
            frames.append(frame)
    corpus = pd.concat(frames, ignore_index=True)
    destination = Path('data/official_kinnews.csv')
    corpus.to_csv(destination, index=False)
    provenance = {
        'source': 'https://raw.githubusercontent.com/saradhix/kinnews_kirnews/master/KINNEWS.zip',
        'dataset_card': 'https://huggingface.co/datasets/andreniyongabo/kinnews_kirnews',
        'research_repository': 'https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus',
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'csv_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'raw_rows': len(corpus),
        'original_split_rows': [len(frame) for frame in frames],
        'label_conversion': 'Original one-based IDs to English class names; verified against all raw en_label values, normalizing politic to politics.',
        'method': 'Pool original cleaned splits; deduplicate and re-split 70/15/15 with seed 42. Not the published benchmark split.',
    }
    Path('data/source.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
