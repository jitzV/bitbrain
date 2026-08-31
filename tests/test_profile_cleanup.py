import os
from types import SimpleNamespace

from profile_store import cleanup_chroma_collection


class FakeClient:
    def __init__(self):
        self.deleted = []

    def get_collection(self, name):
        return SimpleNamespace(id=f"uuid-{name}")

    def delete_collection(self, name):
        self.deleted.append(name)


def test_cleanup_removes_on_disk_collection_storage(tmp_path):
    db_path = tmp_path / "chroma_db"
    db_path.mkdir()
    stale_dir = db_path / "uuid-trust"
    stale_dir.mkdir()
    legacy_dir = db_path / "trust"
    legacy_dir.mkdir()

    client = FakeClient()

    cleanup_chroma_collection(client, "trust", str(db_path))

    assert client.deleted == ["trust"]
    assert not stale_dir.exists()
    assert not legacy_dir.exists()
