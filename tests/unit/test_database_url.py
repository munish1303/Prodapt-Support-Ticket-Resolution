from app.core.database import normalize_database_url


def test_local_asyncpg_url_is_unchanged():
    url = "postgresql+asyncpg://support:support@db:5432/support"
    assert normalize_database_url(url) == (url, {})


def test_hosted_url_is_converted_for_asyncpg():
    url, args = normalize_database_url(
        "postgresql://u:p@ep-cool-name.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    )
    assert url == "postgresql+asyncpg://u:p@ep-cool-name.eu-central-1.aws.neon.tech/neondb?ssl=require"
    assert args == {}


def test_postgres_scheme_alias():
    url, _ = normalize_database_url("postgres://u:p@host:5432/db")
    assert url == "postgresql+asyncpg://u:p@host:5432/db"


def test_pooled_endpoint_disables_prepared_statement_caches():
    url, args = normalize_database_url("postgresql://u:p@ep-x-pooler.eu-central-1.aws.neon.tech/neondb?sslmode=require")
    assert args == {"statement_cache_size": 0}
    assert url.endswith("?ssl=require&prepared_statement_cache_size=0")
