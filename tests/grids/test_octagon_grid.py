import numpy as np


def test_face_areas(ref_octagon):
    face_areas_known = np.full_like(ref_octagon.face_areas, np.sqrt(2) - 1)
    face_areas_known[8:] = 1 / (2 + np.sqrt(2))

    assert np.allclose(ref_octagon.face_areas, face_areas_known)


def test_cell_volumes(ref_octagon):
    cell_volumes_known = np.full_like(ref_octagon.cell_volumes, 0.75 - 1 / np.sqrt(2))
    cell_volumes_known[0] = 2 * np.sqrt(2) - 2

    assert np.allclose(ref_octagon.cell_volumes, cell_volumes_known)
