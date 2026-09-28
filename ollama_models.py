import subprocess


def is_ollama_embedding_error(message):
    """Return True when Ollama is configured without embedding support."""
    if not message:
        return False

    normalized = str(message).lower()
    if "this server does not support embeddings" in normalized:
        return True
    if "--embeddings" in normalized and "embeddings" in normalized:
        return True
    if "status code: 501" in normalized and "embeddings" in normalized:
        return True
    return False


def parse_ollama_list(output):
    """Extract model names from the first column of `ollama list` output."""
    models = []
    for line in output.splitlines():
        columns = line.split()
        if columns and columns[0].upper() != "NAME":
            models.append(columns[0])
    return models


def list_ollama_models():
    """Return locally installed Ollama model names, or an empty list on failure."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_ollama_list(result.stdout)

def resolve_installed_model(model, available_models):
    """Keep an installed model or resolve an untagged name to its :latest tag."""
    if model in available_models:
        return model
    latest_model = f"{model}:latest"
    if latest_model in available_models:
        return latest_model
    return available_models[0] if available_models else None
