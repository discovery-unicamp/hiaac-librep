def test_simple_imports():
    """This test only check if basic librep's modules can be imported
    """

    # Third-party imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import umap
    import plotly.express as px
    import plotly.graph_objects as go

    # Librep imports
    from librep.utils.dataset import PandasDatasetsIO
    from librep.datasets.har.loaders import (
        KuHar_BalancedView20HzMotionSenseEquivalent,
        MotionSense_BalancedView20HZ,
        ExtraSensorySense_UnbalancedView20HZ,
        CHARM_BalancedView20Hz,
        WISDM_UnbalancedView20Hz,
        UCIHAR_UnbalancedView20Hz
    )
    from librep.datasets.multimodal import PandasMultiModalDataset, TransformMultiModalDataset, WindowedTransform
    from librep.transforms.fft import FFT
    from librep.transforms. stats import StatsTransform
    from librep.utils.workflow import SimpleTrainEvalWorkflow, MultiRunWorkflow
    from librep.estimators import RandomForestClassifier, SVC, KNeighborsClassifier
    from librep.metrics.report import ClassificationReport
    from librep.transforms.resampler import SimpleResampler

    assert True
