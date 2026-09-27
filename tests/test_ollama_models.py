import subprocess
from unittest.mock import patch

from ollama_models import list_ollama_models, parse_ollama_list


def test_parse_ollama_list_extracts_names_from_first_column():
    output = (
        "NAME                                               ID              SIZE\n"
        "gemma3:latest                                     123456          4 GB\n"
        "hf.co/example/model:Q4_K_M                       abcdef          2 GB\n"
    )

    assert parse_ollama_list(output) == [
        "gemma3:latest",
        "hf.co/example/model:Q4_K_M",
    ]


def test_list_ollama_models_runs_cli_and_returns_models():
    completed = subprocess.CompletedProcess(
        args=["ollama", "list"], returncode=0, stdout="NAME ID\nmodel:latest abc\n"
    )
    with patch("ollama_models.subprocess.run", return_value=completed) as run:
        assert list_ollama_models() == ["model:latest"]

    run.assert_called_once_with(
        ["ollama", "list"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )


def test_list_ollama_models_returns_empty_when_cli_is_unavailable():
    with patch(
        "ollama_models.subprocess.run", side_effect=FileNotFoundError
    ):
        assert list_ollama_models() == []
