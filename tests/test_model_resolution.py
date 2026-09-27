from ollama_models import resolve_installed_model


def test_resolve_installed_model_keeps_exact_installed_tag():
    installed = ["embed:latest", "vision:q4"]

    assert resolve_installed_model("vision:q4", installed) == "vision:q4"


def test_resolve_installed_model_maps_untagged_name_to_latest():
    installed = ["embed:latest", "vision:q4"]

    assert resolve_installed_model("embed", installed) == "embed:latest"


def test_resolve_installed_model_falls_back_when_model_is_not_installed():
    installed = ["embed:latest", "vision:q4"]

    assert resolve_installed_model("missing", installed) == "embed:latest"
