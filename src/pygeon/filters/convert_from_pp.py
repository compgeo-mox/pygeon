import porepy as pp
import pygeon as pg

def convert_from_pp(obj):
    if isinstance(obj, pg.Graph):
        pass
    elif isinstance(obj, pp.Grid):
        obj.__class__ = pg.Grid
    elif isinstance(obj, pp.MortarGrid):
        obj.__class__ = pg.MortarGrid
    elif isinstance(obj, pp.MixedDimensionalGrid):
        [convert_from_pp(sd) for sd in obj.subdomains()]
        [convert_from_pp(intf) for intf in obj.interfaces()]
        obj.__class__ = pg.MixedDimensionalGrid
    else:
        raise TypeError
