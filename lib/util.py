class LimitedStore:
    def __init__(self, size):
        self.max_size = size
        self.values = []
    
    def size(self):
        return len(self.values)
    
    def add(self, val):
        if len(self.values) == self.max_size:
            self.values.pop(0)
        self.values.append(val)

        
import os
from lib.schema import Session, Protein, engine
import pandas as pd

def find_contacts(file_path):
    result = os.popen(f'~/cs590/bin/confind --p {file_path} --rLib ~/cs590/bin/rotlibs').read()

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
    
    return list(zip(contact_x, contact_y))
#     protein.contact_x = contact_x
#     protein.contact_y = contact_y
#     session.commit()
#     print("DONE", pdb_id)
    