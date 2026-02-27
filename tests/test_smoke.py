def test_smoke_imports() -> None:
    import faiss  # noqa: F401
    import fastapi  # noqa: F401
    from langchain_community.vectorstores import FAISS  # noqa: F401
    from mistralai import Mistral  # noqa: F401
    from langchain_mistralai.embeddings import MistralAIEmbeddings  # noqa: F401
