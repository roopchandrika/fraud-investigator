"""Thin data-access layer for Postgres and Neo4j (no ORM on purpose: keep the SQL visible)."""
import atexit
from contextlib import contextmanager
from functools import lru_cache

import psycopg
from neo4j import Driver, GraphDatabase
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from fraud_agent.config import get_settings


@lru_cache
def pg_pool() -> ConnectionPool:
    pool = ConnectionPool(get_settings().pg_dsn, min_size=1, max_size=10,
                          kwargs={"row_factory": dict_row})
    atexit.register(pool.close)
    return pool


@contextmanager
def pg() -> psycopg.Connection:
    with pg_pool().connection() as conn:
        yield conn


@lru_cache
def neo4j() -> Driver:
    s = get_settings()
    return GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))


def init_neo4j() -> None:
    with neo4j().session() as session:
        session.run("CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE")
        session.run("CREATE CONSTRAINT merchant_id IF NOT EXISTS FOR (m:Merchant) REQUIRE m.id IS UNIQUE")
