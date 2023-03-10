import tempfile
from librep.datasets.har.loaders import (
    KuHar_BalancedView20HzMotionSenseEquivalent,
    MotionSense_BalancedView20HZ,
    ExtraSensorySense_UnbalancedView20HZ,
    WISDM_BalancedView20Hz,
    UCIHAR_BalancedView20Hz,
    MegaHARDataset_BalancedView20Hz
)
import pytest

def download_load(loader_cls: type, min_samples: int = 10):
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = loader_cls(root_dir=tmpdir, download=True)
        dset = loader.load(concat_all=True)
        assert len(dset) > min_samples, "Dataset is empty"

@pytest.mark.skip(reason="slow test")
def test_kuhar_download_load():
    download_load(KuHar_BalancedView20HzMotionSenseEquivalent)

@pytest.mark.skip(reason="slow test")
def test_motionsense_download_load():
    download_load(MotionSense_BalancedView20HZ)

@pytest.mark.skip(reason="slow test")
def test_extrasensory_download_load():
    download_load(ExtraSensorySense_UnbalancedView20HZ)

@pytest.mark.skip(reason="slow test")
def test_wisdm_download_load():
    download_load(WISDM_BalancedView20Hz)

@pytest.mark.skip(reason="slow test")
def test_ucihar_download_load():
    download_load(UCIHAR_BalancedView20Hz)

@pytest.mark.skip(reason="slow test")
def test_megahar_download_load():
    download_load(MegaHARDataset_BalancedView20Hz)
