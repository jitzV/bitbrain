import os
import shutil
import sqlite3


def _collection_storage_names(client, collection_name, db_path):
    names = {collection_name}

    try:
        collection = client.get_collection(collection_name)
        collection_id = getattr(collection, "id", None)
        if collection_id:
            names.add(collection_id)
    except Exception:
        pass

    sqlite_path = os.path.join(db_path, "chroma.sqlite3")
    try:
        with sqlite3.connect(sqlite_path) as conn:
            rows = conn.execute(
                "SELECT id, name FROM collections WHERE name = ? OR id = ?",
                (collection_name, collection_name),
            ).fetchall()
            for collection_id, collection_name_row in rows:
                names.add(collection_id)
                names.add(collection_name_row)
    except Exception:
        pass

    return names


def cleanup_chroma_collection(client, collection_name, db_path):
    """Delete a Chroma collection and any leftover collection folders from local storage."""
    if not db_path:
        return

    storage_names = _collection_storage_names(client, collection_name, db_path)

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    try:
        entries = os.listdir(db_path)
    except FileNotFoundError:
        return

    for entry in entries:
        full_path = os.path.join(db_path, entry)
        if not os.path.isdir(full_path):
            continue

        if entry in storage_names or entry.startswith(f"{collection_name}-"):
            shutil.rmtree(full_path, ignore_errors=True)
