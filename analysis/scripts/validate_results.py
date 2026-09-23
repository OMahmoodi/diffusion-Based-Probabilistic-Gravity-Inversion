from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parents[2]
overall=pd.read_csv(root/'analysis/data/overall_summary.csv')
paired=pd.read_csv(root/'analysis/data/paired_statistical_tests.csv')
assert set(overall['split'])=={'ID','OOD'}
assert overall.groupby('split').size().to_dict()=={'ID':4,'OOD':4}
assert len(paired)==12
print('Validated frozen analysis tables:', len(overall), 'summary rows and', len(paired), 'paired tests.')
