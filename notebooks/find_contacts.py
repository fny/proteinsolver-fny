import os
import project_path
from lib.schema import Session, Protein, engine
import pandas as pd
import random
from sqlalchemy.sql import text

session = Session()
pdb_ids = list(map(lambda x: x[0], session.query(Protein.pdb_id).filter_by(contact_x = None).all()))
random.shuffle(pdb_ids)

def find_contacts(pdb_id):
    protein = session.query(Protein).filter_by(pdb_id=pdb_id).first()
    if protein.contact_x is not None:
        print("DIDZ", pdb_id)
        return
    result = os.popen(f'~/cs590/bin/confind --p {protein.file_path} --rLib ~/cs590/bin/rotlibs').read()

    data = []
    for line in result.split('\n'):
        if line.startswith('contact'):
            data.append(line.split('\t'))


    # pd.set_option('display.max_rows', 5000)

    df = pd.DataFrame(data).sort_values(by=[3], ascending=False)
    df[3] = pd.to_numeric(df[3],errors='coerce')
    df = df[df[3] > 0.5]

    contact_x = list(map(lambda x: int(x.split(',')[1]), df[df[3] > 0.5][1]))
    contact_y = list(map(lambda x: int(x.split(',')[1]), df[df[3] > 0.5][2]))
    protein.contact_x = contact_x
    protein.contact_y = contact_y
    session.commit()
    print("DONE", pdb_id)
    
for pdb_id in pdb_ids:
    find_contacts(pdb_id)