"""Mathematical and raster invariants for the isolated normalization experiment."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from threadpoolctl import threadpool_limits

from irmad_rrn import OUT, Moments, fit_irmad, ncp, tls, write_normalized


class NormalizationTests(unittest.TestCase):
    def test_changed_population_and_tls_recovery(self):
        rng = np.random.default_rng(901)
        target = rng.uniform(30, 160, (40000, 3))
        expected_gain, expected_offset = np.array([.8, 1.1, .9]), np.array([12, -8, 4])
        reference = target*expected_gain+expected_offset+rng.normal(0, .6, target.shape)
        reference[:8000] = rng.uniform(20, 200, (8000, 3))
        z = np.column_stack([reference, target])
        with threadpool_limits(limits=1):
            model, _, _, _ = fit_irmad(z)
            pif = ncp(z, model)>.95
            self.assertLess(pif[:8000].mean(), pif[8000:].mean()*.03)
            m = Moments(6)
            m.add(z[pif])
            gain, offset, _ = tls(*m.result())
        np.testing.assert_allclose(gain, expected_gain, atol=.003)
        np.testing.assert_allclose(offset, expected_offset, atol=.3)

    def test_metadata_nodata_mask_and_no_overwrite(self):
        OUT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUT) as temp:
            source, dest = Path(temp)/'raw.tif', Path(temp)/'normalized.tif'
            values = np.full((3, 32, 32), 100, dtype='uint8')
            values[:, 0, 0] = 255
            values[1, 0, 1] = 255  # Per-band nodata must not become a shared mask.
            with rasterio.open(source, 'w', driver='GTiff', width=32, height=32, count=3,
                               dtype='uint8', nodata=255, crs='EPSG:4326', transform=from_origin(114, 24, .001, .001)) as ds:
                ds.write(values)
                ds.update_tags(AREA_OR_POINT='Area', custom='preserved')
                ds.set_band_description(1, 'Red')
            write_normalized(source, dest, np.ones(3)*3, np.zeros(3))
            with rasterio.open(dest) as ds:
                self.assertEqual(ds.tags()['custom'], 'preserved')
                self.assertEqual(ds.descriptions[0], 'Red')
                self.assertEqual(ds.read()[0, 1, 1], 254)
                self.assertEqual(ds.read()[0, 0, 0], 255)
            with self.assertRaises(FileExistsError):
                write_normalized(source, dest, np.ones(3), np.zeros(3))


if __name__ == '__main__':
    unittest.main()
