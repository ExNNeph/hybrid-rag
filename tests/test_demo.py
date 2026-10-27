import builtins

import pytest
from streamlit.testing.v1 import AppTest

from hybrid_rag import answer as qa
from hybrid_rag import cli
from tests.test_answer import fake_llm


@pytest.fixture
def app():
    return AppTest.from_file(str(cli.DEMO_APP), default_timeout=30)  # the file `hybrid-rag demo` starts


def test_retrieval_only_without_key(app):
    at = app.run()
    assert not at.exception
    at.text_input(key="q-None").input("What does API error TB-4031 mean?").run()
    at.button[0].click().run()
    assert not at.exception and any("retrieval only" in m.value for m in at.markdown)
    assert len(at.expander) >= 4  # top chunks from each retriever


def test_answer_cache_hit_and_trap(app, monkeypatch):
    monkeypatch.setattr(qa, "_llm", fake_llm)
    at = app.run()
    at.sidebar.text_input[0].input("x").run()  # the key
    for q in ("Can I get a refund on an annual plan?", "Can I get a refund on an annual plan",
              "Can I get a refund on a monthly plan?"):
        at.text_input(key="q-None").input(q).run()
        at.button[0].click().run()
        assert not at.exception
        if not q.endswith("?"):
            assert any("Cache hit" in s.value for s in at.success)
    assert any("Cache miss" in i.value and "annual plan?" in i.value for i in at.info)  # names the closest
    assert any("[1]" in m.value for m in at.markdown)


def test_demo_without_streamlit_gives_install_hint(monkeypatch):
    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__",
                        lambda n, *a, **k: (_ for _ in ()).throw(ImportError()) if n == "streamlit" else real(n, *a, **k))
    with pytest.raises(SystemExit, match="pip install"):
        cli.demo()
