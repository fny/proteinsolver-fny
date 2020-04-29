from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, Boolean, Float, Text, PickleType
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.declarative import declarative_base

from . import protein_mixin

engine = create_engine('postgresql://cs590:password@linux15:5432/cs590', pool_size=30)


Base = declarative_base()

class Protein(Base, protein_mixin.ProteinMixin):
    __tablename__ = 'proteins'    
    pdb_id = Column(String, primary_key=True)
    has_discont = Column(Boolean)
    pdb_data = Column(Text)
    pdb_name = Column(String)
    # pdb_header = Column(JSONB)
    family = Column(String, index=True)
    is_clean = Column(Boolean)
    has_missing_residues = Column(Boolean)
    sequence = Column(String)
    length = Column(Integer)
    phi_angles = Column(ARRAY(Float))
    psi_angles = Column(ARRAY(Float))
    file_path = Column(String)
    
    def __repr__(self):
        return f"<Protein(pdb_id='{self.pdb_id}', family='{self.family}', length={self.length})>"

from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)
    
Base.metadata.create_all(engine)