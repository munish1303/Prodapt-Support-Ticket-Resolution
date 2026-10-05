from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_vercel_dockerfile_matches_dockerfile():
    # Vercel builds Dockerfile.vercel; keeping it identical means the hosted image is exactly the tested one.
    assert (ROOT / "Dockerfile.vercel").read_bytes() == (ROOT / "Dockerfile").read_bytes()
