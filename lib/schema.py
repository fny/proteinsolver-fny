from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, Boolean, Float, Text, PickleType
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.declarative import declarative_base

from . import protein_mixin

engine = create_engine('postgresql://cs590:password@linux59:5432/cs590', pool_size=30)


Base = declarative_base()

class Protein(Base, protein_mixin.ProteinMixin):
    __tablename__ = 'proteins'    
    pdb_id = Column(String, primary_key=True)
    has_discont = Column(Boolean)
    pdb_data = Column(Text)
    pdb_name = Column(String)
    family = Column(String, index=True)
    is_clean = Column(Boolean)
    has_missing_residues = Column(Boolean)
    sequence = Column(String)
    length = Column(Integer)
    phi_angles = Column(ARRAY(Float))
    psi_angles = Column(ARRAY(Float))
    file_path = Column(String)
    contact_x = Column(ARRAY(Integer))
    contact_y = Column(ARRAY(Integer))
    
    def __repr__(self):
        return f"<Protein(pdb_id='{self.pdb_id}', family='{self.family}', length={self.length})>"

class ProteinSolverResult(Base):
    __tablename__ = "protein_solver_results"
    pdb_id = Column(String, primary_key=True)

    sequences = Column(ARRAY(String))
    log_prob_sums = Column(ARRAY(Float))
    log_prob_avgs = Column(ARRAY(Float))
    seq_identities = Column(ARRAY(Float))
    max_prob_avg = Float
    n_results = Integer
    def __repr__(self):
        return f"<PSResult(pdb_id='{self.pdb_id}'>"
    
from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)
    
Base.metadata.create_all(engine)