from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/"src")]
from rdflib import Graph,Namespace,RDF,RDFS,OWL
from src.ocosl_constraints_final import build_class_index,is_subclass_or_same
def test_subclass_closure():
    g=Graph(); ex=Namespace("http://x#")
    for c in [ex.PhysicalObject,ex.Vehicle,ex.Boat]: g.add((c,RDF.type,OWL.Class))
    g.add((ex.Vehicle,RDFS.subClassOf,ex.PhysicalObject)); g.add((ex.Boat,RDFS.subClassOf,ex.Vehicle))
    idx=build_class_index(g)
    assert is_subclass_or_same(g,idx,"Boat","Vehicle")
    assert is_subclass_or_same(g,idx,"Boat","PhysicalObject")
