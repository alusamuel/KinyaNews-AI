import pandas as pd
from src.prepare_data import clean_frame
from src.labels import name,kin_name

def test_label_mapping():
    assert name(1)=='sport'
    assert name('12')=='education'
    assert kin_name(1)=='Imikino'

def test_exact_duplicate_conflict_removal():
    text='amakuru menshi yerekeye umukino w umupira muri kigali'
    df=pd.DataFrame({'title':[text]*3+['Amakuru mashya yerekeye uburezi mu Rwanda'], 'content':['']*4,'label':[1,1,2,12]})
    # After contradictory annotations are discarded, single education row remains, so minimum support guard triggers.
    import pytest
    with pytest.raises(ValueError,match='Too few rows'):
        clean_frame(df)
