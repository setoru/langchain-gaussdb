from __future__ import annotations

import json
import uuid
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Iterable, Optional, Sequence, List, Tuple, Callable

from psycopg_pool import ConnectionPool, AsyncConnectionPool
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .config import GaussVectorSettings


class GaussVectorStore(VectorStore):
    """GaussDB store integration.

    Setup:
        Install ``langchain-gaussdb`` and deploy a standalone GaussDB server.

    Key init args — core params:
        embedding: Embeddings
            Function used to embed the text.
        config: GaussVectorSettings
            Configuration settings for GaussDB connection and indexing

    Instantiate:
        .. code-block:: python

            from langchain_gaussdb import GaussVectorStore, GaussVectorSettings
            from langchain_openai import OpenAIEmbeddings

            config = GaussVectorSettings(
                host="localhost",
                port=5432,
                user="gaussdb",
                password="Test@123456",
                database="postgres",
                table_name="my_docs"
            )
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large"
            )
            vector_store = GaussVectorStore(
                embedding=embeddings,
                config=config
            )

    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(")

            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud", k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * thud [{'bar': 'baz'}]

    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud", k=1, filter={"bar": "baz"})
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * thud [{'bar': 'baz'}]

    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux", k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.559320] foo [{'baz': 'bar'}]

    Table management:
            .. code-block:: python

                # Create new collection/table
                vector_store._create_table()

                # Drop entire table
                vector_store.drop_table()

    Async:
        .. code-block:: python

            # add documents
            await vector_store.aadd_documents(documents=documents, ids=ids)

            # delete documents
            await vector_store.adelete(ids=["3"])

            # search
            results = vector_store.asimilarity_search(query="thud",k=1)

            # search with score
            # results = await vector_store.asimilarity_search_with_score(query="qux",k=1)
            for doc,score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.499243] foo [{'baz': 'bar'}]

    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 1},
            )
            retriever.invoke("thud")

        .. code-block:: python

            [Document(metadata={'bar': 'baz'}, page_content='thud')]

    """
    def __init__(self, embedding: Embeddings, config: GaussVectorSettings):
        """Initialize the GaussVector store"""
        self.embedding_function = embedding
        self.config = config
        self.pool = None
        self.async_pool = None
        self._init_pool()
        self._create_table()

    def _init_pool(self) -> None:
        """Initialize connection pool"""
        conn_info = (
            f"host={self.config.host} port={self.config.port} "
            f"user={self.config.user} password={self.config.password} "
            f"dbname={self.config.database}"
        )
        self.pool = ConnectionPool(
            conninfo=conn_info,
            min_size=self.config.min_connections,
            max_size=self.config.max_connections,
            open=False,
        )
        self.pool.open()

    @contextmanager
    def _get_cursor(self):
        """Get database cursor with context management"""
        with self.pool.connection() as conn:
            try:
                with conn.cursor() as cur:
                    yield cur
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    async def _init_async_pool(self) -> None:
        conn_info = (
            f"host={self.config.host} port={self.config.port} "
            f"user={self.config.user} password={self.config.password} "
            f"dbname={self.config.database}"
        )
        self.async_pool = AsyncConnectionPool(
            conninfo=conn_info,
            min_size=self.config.min_connections,
            max_size=self.config.max_connections,
            open=False,
        )
        await self.async_pool.open()

    @asynccontextmanager
    async def _get_async_cursor(self):
        async with self.async_pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    yield cur
                await conn.commit()
            except Exception as e:
                await conn.rollback()
                raise e

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    async def aclose(self):
        if self.async_pool is not None:
            await self.async_pool.close()
            self.async_pool = None

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select the relevance score function based on the distance strategy."""
        if self.config.distance_strategy.value == "cosine":
            return self._cosine_relevance_score_fn
        elif self.config.distance_strategy.value == "l2":
            return self._euclidean_relevance_score_fn
        elif self.config.distance_strategy.value == "inner_product":
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                f"No supported normalization function for distance_strategy of {self.config.distance_strategy.value}."
            )

    def _index_exists(self, index_name: str) -> bool:
        """Check if index exists in the database"""
        check_sql = """
            SELECT EXISTS (
                SELECT 1 
                FROM pg_indexes 
                WHERE tablename = %s AND indexname = %s
            );
        """
        with self._get_cursor() as cur:
            cur.execute(check_sql, (self.config.table_name, index_name))
            return cur.fetchone()[0]

    def _create_table(self) -> None:
        """Create table with vector index if not exists"""
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata JSONB,
                embedding {self.config.vector_type.value}({self.config.embedding_dimension}) NOT NULL
            );
        """
        with self._get_cursor() as cur:
            cur.execute(create_table_sql)

        index_name = f"idx_{self.config.table_name}_embedding"
        if not self._index_exists(index_name):
            self.create_index(index_name=index_name, drop_if_exists=False)

    def create_index(self, index_name=None, drop_if_exists=True) -> None:
        """Create vector index for the table"""
        if index_name is None:
            index_name = f"idx_{self.config.table_name}_embedding"
        if drop_if_exists:
            drop_sql = f"DROP INDEX IF EXISTS {index_name};"
            with self._get_cursor() as cur:
                cur.execute(drop_sql)

        params_dict = self.config.index_params.model_dump()
        with_clause = ", ".join([f"{key} = {value}" for key, value in params_dict.items()])

        create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {self.config.table_name}
            USING {self.config.index_type.value} (embedding {self.config.distance_strategy.value})
            WITH ({with_clause});
        """
        with self._get_cursor() as cur:
            cur.execute(create_index_sql)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadata associated with the texts.
            ids: Optional list of IDs associated with the texts.
            **kwargs: vectorstore specific parameters.

        Returns:
            List of ids from adding the texts into the vectorstore.

        Raises:
            ValueError: If the number of metadatas does not match the number of texts.
            ValueError: If the number of ids does not match the number of texts.
        """
        texts = list(texts)
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        elif len(metadatas) != len(texts):
            msg = (
                "The number of metadatas must match the number of texts."
                f"Got {len(metadatas)} metadatas and {len(texts)} texts."
            )
            raise ValueError(msg)

        if ids is None:
            generated_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        elif len(ids) != len(texts):
            msg = (
                "The number of ids must match the number of texts."
                f"Got {len(ids)} ids and {len(texts)} texts."
            )
            raise ValueError(msg)
        else:
            # Create a copy to avoid issues with list object modifications
            generated_ids = list(ids)

        embeddings = self.embedding_function.embed_documents(texts)
        records = []

        for doc_id, text, metadata, embedding in zip(generated_ids, texts, metadatas, embeddings):
            records.append((
                doc_id,
                text,
                json.dumps(metadata),
                json.dumps(embedding)
            ))

        insert_sql = f"""
            INSERT INTO {self.config.table_name} (id, text, metadata, embedding)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE text = VALUES(text), metadata = VALUES(metadata), embedding = VALUES(embedding);
        """
        with self._get_cursor() as cur:
            cur.executemany(insert_sql, records)
        return generated_ids

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Add or update documents in the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            kwargs: Additional keyword arguments.
                if kwargs contains ids and documents contain ids,
                the ids in the kwargs will receive precedence.

        Returns:
            List of IDs of the added texts.

        Raises:
            ValueError: If the number of ids does not match the number of documents.
        """
        metadatas = []
        texts = []

        for doc in documents:
            metadatas.append(doc.metadata)
            texts.append(doc.page_content)
        if 'ids' not in kwargs:
            # Fill missing IDs with UUIDs
            ids = []
            for i, doc in enumerate(documents):
                if hasattr(doc, 'id') and doc.id is not None:
                    ids.append(doc.id)
                else:
                    ids.append(str(uuid.uuid4()))
            kwargs['ids'] = tuple(ids)
        else:
            if len(kwargs['ids']) != len(documents):
                msg = (
                    "The number of ids must match the number of documents."
                    f"Got {len(kwargs['ids'])} ids and {len(documents)} documents."
                )
                raise ValueError(msg)

        return self.add_texts(texts, metadatas, **kwargs)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any
    ) -> list[str]:
        """Async run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadata associated with the texts.
            ids: Optional list of IDs associated with the texts.
            **kwargs: vectorstore specific parameters.

        Returns:
            List of ids from adding the texts into the vectorstore.

        Raises:
            ValueError: If the number of metadatas does not match the number of texts.
            ValueError: If the number of ids does not match the number of texts.
        """
        texts = list(texts)
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        elif len(metadatas) != len(texts):
            msg = (
                "The number of metadatas must match the number of texts."
                f"Got {len(metadatas)} metadatas and {len(texts)} texts."
            )
            raise ValueError(msg)

        if ids is None:
            generated_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        elif len(ids) != len(texts):
            msg = (
                "The number of ids must match the number of texts."
                f"Got {len(ids)} ids and {len(texts)} texts."
            )
            raise ValueError(msg)
        else:
            # Create a copy to avoid issues with list object modifications
            generated_ids = list(ids)

        embeddings = self.embedding_function.embed_documents(texts)
        records = []

        for doc_id, text, metadata, embedding in zip(generated_ids, texts, metadatas, embeddings):
            records.append((
                doc_id,
                text,
                json.dumps(metadata),
                json.dumps(embedding)
            ))

        insert_sql = f"""
            INSERT INTO {self.config.table_name} (id, text, metadata, embedding)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE text = VALUES(text), metadata = VALUES(metadata), embedding = VALUES(embedding);
        """
        await self._init_async_pool()
        async with self._get_async_cursor() as cur:
            await cur.executemany(insert_sql, records)

        return generated_ids

    async def aadd_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Async add or update documents in the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            kwargs: Additional keyword arguments.
                if kwargs contains ids and documents contain ids,
                the ids in the kwargs will receive precedence.

        Returns:
            List of IDs of the added texts.

        Raises:
            ValueError: If the number of ids does not match the number of documents.
        """
        metadatas = []
        texts = []

        for doc in documents:
            metadatas.append(doc.metadata)
            texts.append(doc.page_content)
        if 'ids' not in kwargs:
            # Fill missing IDs with UUIDs
            ids = []
            for i, doc in enumerate(documents):
                if hasattr(doc, 'id') and doc.id is not None:
                    ids.append(doc.id)
                else:
                    ids.append(str(uuid.uuid4()))
            kwargs['ids'] = tuple(ids)
        else:
            if len(kwargs['ids']) != len(documents):
                msg = (
                    "The number of ids must match the number of documents."
                    f"Got {len(kwargs['ids'])} ids and {len(documents)} documents."
                )
                raise ValueError(msg)

        generated_ids = await self.aadd_texts(texts, metadatas, **kwargs)
        return generated_ids

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete. If None, delete all. Default is None.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        with self._get_cursor() as cur:
            if ids is None:
                delete_sql = f"DELETE FROM {self.config.table_name}"
                cur.execute(delete_sql)
            else:
                delete_sql = f"DELETE FROM {self.config.table_name} WHERE id = ANY(%s)"
                cur.execute(delete_sql, (ids,))
            return True

    async def adelete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Async delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete. If None, delete all. Default is None.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        await self._init_async_pool()
        async with self._get_async_cursor() as cur:
            if ids is None:
                delete_sql = f"DELETE FROM {self.config.table_name}"
                await cur.execute(delete_sql)
            else:
                delete_sql = f"DELETE FROM {self.config.table_name} WHERE id = ANY(%s)"
                await cur.execute(delete_sql, (ids,))
            return True

    def get_by_ids(self, ids: Sequence[str]) -> list[Document]:
        """Get documents by their IDs.

        Args:
            ids: List of ids to retrieve.

        Returns:
            List of Documents.
        """
        with self._get_cursor() as cur:
            query = f"SELECT id, text, metadata FROM {self.config.table_name} WHERE id = ANY(%s)"
            cur.execute(query, (list(ids),))
            docs = []
            for _id, text, metadata in cur:
                docs.append(
                    Document(id=_id, page_content=text, metadata=metadata)
                )
        return docs

    async def aget_by_ids(self, ids: Sequence[str]) -> list[Document]:
        """Async get documents by their IDs.

        Args:
            ids: List of ids to retrieve.

        Returns:
            List of Documents.
        """
        await self._init_async_pool()
        async with self._get_async_cursor() as cur:
            query = f"SELECT id, text, metadata FROM {self.config.table_name} WHERE id = ANY(%s)"
            await cur.execute(query, (list(ids),))
            docs = []
            async for _id, text, metadata in cur:
                docs.append(
                    Document(id=_id, page_content=text, metadata=metadata)
                )
        return docs

    def search(self, query: str, search_type: str, **kwargs: Any) -> list[Document]:
        """Return docs most similar to query using a specified search type.

        Args:
            query: Input text
            search_type: Type of search to perform. Can be "similarity",
                "mmr", or "similarity_score_threshold".
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.

        Raises:
            ValueError: If search_type is not supported.
        """
        if search_type == "similarity":
            return self.similarity_search(query, **kwargs)
        elif search_type == "similarity_score_threshold":
            docs_and_similarities = self.similarity_search_with_relevance_scores(query, **kwargs)
            return [doc for doc, _ in docs_and_similarities]
        else:
            raise ValueError(f"Search type {search_type} is not supported yet.")

    async def asearch(self, query: str, search_type: str, **kwargs: Any) -> list[Document]:
        """Async return docs most similar to query using a specified search type.

        Args:
            query: Input text.
            search_type: Type of search to perform. Can be "similarity",
                "mmr", or "similarity_score_threshold".
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.

        Raises:
            ValueError: If search_type is not one of "similarity",
                "mmr", or "similarity_score_threshold".
        """
        if search_type == "similarity":
            return await self.asimilarity_search(query, **kwargs)
        elif search_type == "similarity_score_threshold":
            docs_and_similarities = await self.asimilarity_search_with_relevance_scores(query, **kwargs)
            return [doc for doc, _ in docs_and_similarities]
        else:
            raise ValueError(f"Search type {search_type} is not supported yet.")

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        query_vector = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector(
            embedding=query_vector, k=k, filter=filter
        )

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async return docs most similar to query.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        query_vector = await self.embedding_function.aembed_query(query)
        return await self.asimilarity_search_by_vector(
            embedding=query_vector, k=k, filter=filter
        )


    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and scores most similar to query.

        Args:
            query: Input text.
            k: Number of results to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.

        Returns:
            List of (Document, score) tuples most similar to the query.
        """
        query_vector = self.embedding_function.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding=query_vector, k=k, filter=filter
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and scores most similar to query.

        Args:
            query: Input text.
            k: Number of results to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.

        Returns:
            List of (Document, score) tuples most similar to the query.
        """
        query_vector = await self.embedding_function.aembed_query(query)
        return await self.asimilarity_search_with_score_by_vector(
            embedding=query_vector, k=k, filter=filter
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ):
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and scores most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of results to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.

        Returns:
            List of (Document, score) tuples most similar to the query vector.
        """
        with self._get_cursor() as cur:
            if filter is None:
                sql = f"""
                    SELECT id, text, metadata, embedding {self.config.index_operator} %s AS distance
                    FROM {self.config.table_name}
                    ORDER BY distance LIMIT %s
                """
                cur.execute(sql, (json.dumps(embedding), k))
            else:
                sql = f"""
                    SELECT id, text, metadata, embedding {self.config.index_operator} %s AS distance
                    FROM {self.config.table_name}
                    WHERE metadata @> %s
                    ORDER BY distance LIMIT %s
                """
                cur.execute(sql, (json.dumps(embedding), json.dumps(filter), k))
            results = []
            for _id, text, metadata, distance in cur:
                results.append((
                    Document(
                        id=_id,
                        page_content=text,
                        metadata=metadata
                    ),
                    distance
                ))
        return results

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and scores most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of results to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.

        Returns:
            List of (Document, score) tuples most similar to the query vector.
        """
        await self._init_async_pool()
        async with self._get_async_cursor() as cur:
            if filter is None:
                sql = f"""
                    SELECT id, text, metadata, embedding {self.config.index_operator} %s AS distance
                    FROM {self.config.table_name}
                    ORDER BY distance LIMIT %s
                """
                await cur.execute(sql, (json.dumps(embedding), k))
            else:
                sql = f"""
                    SELECT id, text, metadata, embedding {self.config.index_operator} %s AS distance
                    FROM {self.config.table_name}
                    WHERE metadata @> %s
                    ORDER BY distance LIMIT %s
                """
                await cur.execute(sql, (json.dumps(embedding), json.dumps(filter), k))
            results = []
            async for _id, text, metadata, distance in cur:
                results.append((
                    Document(
                        id=_id,
                        page_content=text,
                        metadata=metadata
                    ),
                    distance
                ))
        return results

    def drop_table(self):
        """Drop table if exists"""
        with self._get_cursor() as cur:
            drop_sql = f"DROP TABLE IF EXISTS {self.config.table_name}"
            cur.execute(drop_sql)

    @classmethod
    def from_documents(
        cls: type[GaussVectorStore],
        documents: list[Document],
        embedding: Embeddings,
        config: Optional[GaussVectorSettings] = None,
        **kwargs: Any,
    ) -> GaussVectorStore:
        """Return VectorStore initialized from documents and embeddings.

        Args:
            documents: List of Documents to add to the vectorstore.
            embedding: Embedding function to use.
            config: OpenGauss settings configuration.
            kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from documents and embeddings.
        """
        if config is None:
            raise ValueError("GaussVectorSettings config is required")

        metadatas = []
        texts = []

        for doc in documents:
            metadatas.append(doc.metadata)
            texts.append(doc.page_content)

        if "ids" not in kwargs:
            ids = []
            for i, doc in enumerate(documents):
                if hasattr(doc, 'id') and doc.id is not None:
                    ids.append(doc.metadata["id"])
                else:
                    ids.append(str(uuid.uuid4()))
            kwargs['ids'] = tuple(ids)
        else:
            if len(kwargs['ids']) != len(documents):
                msg = (
                    "The number of ids must match the number of documents."
                    f"Got {len(kwargs['ids'])} ids and {len(documents)} documents."
                )
                raise ValueError(msg)

        return cls.from_texts(texts, embedding, metadatas, config=config, **kwargs)

    @classmethod
    def from_texts(
        cls: type[GaussVectorStore],
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        config: Optional[GaussVectorSettings] = None,
        **kwargs: Any,
    ) -> GaussVectorStore:
        """Return VectorStore initialized from texts and embeddings.

        Args:
            texts: Texts to add to the vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
            config: GaussVectorStore settings configuration.
            kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from texts and embeddings.
        """
        if config is None:
            raise ValueError("GaussVectorSettings config is required")

        vector_db = cls(
            embedding=embedding,
            config=config
        )
        vector_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return vector_db

    @classmethod
    async def afrom_documents(
        cls: type[GaussVectorStore],
        documents: list[Document],
        embedding: Embeddings,
        config: Optional[GaussVectorSettings] = None,
        **kwargs: Any,
    ) -> GaussVectorStore:
        """Async return VectorStore initialized from documents and embeddings.

        Args:
            documents: List of Documents to add to the vectorstore.
            embedding: Embedding function to use.
            config: OpenGauss settings configuration.
            kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from documents and embeddings.
        """
        if config is None:
            raise ValueError("GaussVectorSettings config is required")

        metadatas = []
        texts = []

        for doc in documents:
            metadatas.append(doc.metadata)
            texts.append(doc.page_content)

        if "ids" not in kwargs:
            ids = []
            for i, doc in enumerate(documents):
                if hasattr(doc, 'id') and doc.id is not None:
                    ids.append(doc.metadata["id"])
                else:
                    ids.append(str(uuid.uuid4()))
            kwargs['ids'] = tuple(ids)
        else:
            if len(kwargs['ids']) != len(documents):
                msg = (
                    "The number of ids must match the number of documents."
                    f"Got {len(kwargs['ids'])} ids and {len(documents)} documents."
                )
                raise ValueError(msg)

        return await cls.afrom_texts(texts, embedding, metadatas, config=config, **kwargs)

    @classmethod
    async def afrom_texts(
        cls: type[GaussVectorStore],
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        config: Optional[GaussVectorSettings] = None,
        **kwargs: Any,
    ) -> GaussVectorStore:
        """Async return VectorStore initialized from texts and embeddings.

        Args:
            texts: Texts to add to the vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
            config: GaussVectorStore settings configuration.
            kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from texts and embeddings.
        """
        if config is None:
            raise ValueError("GaussVectorSettings config is required")

        vector_db = cls(
            embedding=embedding,
            config=config
        )
        await vector_db.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)
        return vector_db
