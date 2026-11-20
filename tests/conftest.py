import os

os.environ.setdefault("ROUTER_EMBEDDER", "hash")  # tests never download models

import pytest

from router import db as dbm
from router.synthetic import populate


@pytest.fixture(scope="session")
def synth_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "synthetic.db"
    conn = dbm.init_db(path)
    populate(conn, n=900, seed=1)
    conn.close()
    return path
