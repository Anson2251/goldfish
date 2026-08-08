import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))


def _load_main():
    spec = importlib.util.spec_from_file_location("goldfish_info_main", ROOT / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_info_prints_language_profile_summary(capsys) -> None:
    main = _load_main()

    assert main(["info", str(ROOT / "model-profiles" / "language" / "gru-small.yaml")]) == 0

    output = capsys.readouterr().out
    assert "Model:   language/gru" in output
    assert "Total params:" in output
    assert "GRULanguageModel" in output


def test_info_prints_forecast_profile_summary(capsys) -> None:
    main = _load_main()

    assert main(["info", str(ROOT / "model-profiles" / "forecast" / "multihead-lstm-small.yaml")]) == 0

    output = capsys.readouterr().out
    assert "Model:   forecast/multihead-lstm" in output
    assert "Total params:" in output
    assert "MultiHeadLSTMForecastModel" in output


def test_info_prints_conv_lstm_forecast_profile_summary(capsys) -> None:
    main = _load_main()

    assert main(["info", str(ROOT / "model-profiles" / "forecast" / "conv-lstm-small.yaml")]) == 0

    output = capsys.readouterr().out
    assert "Model:   forecast/conv-lstm" in output
    assert "Total params:" in output
    assert "Conv1d" in output
    assert "LSTM" in output


def test_info_prints_linear_lstm_forecast_profile_summary(capsys) -> None:
    main = _load_main()

    assert main(["info", str(ROOT / "model-profiles" / "forecast" / "linear-lstm-small.yaml")]) == 0

    output = capsys.readouterr().out
    assert "Model:   forecast/linear-lstm" in output
    assert "Total params:" in output
    assert "Linear" in output
    assert "LSTM" in output
