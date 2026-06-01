import psycopg2
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


client = TestClient(app)


def get_conn():
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )


def test_latest_scenario_filters_by_family():
    for event_type in ["market_crash", "credit_crisis", "rate_shock", "global_shock"]:
        response = client.get(f"/api/v1/scenarios/latest?event_type={event_type}")
        assert response.status_code == 200
        assert response.json()["family"]["eventType"] == event_type


def test_generate_invalid_family_returns_400():
    response = client.post(
        "/api/v1/scenarios/generate",
        json={
            "family_id": "bogus",
            "severity": "Severe",
            "horizon": 60,
            "displayed_paths": 200,
        },
    )
    assert response.status_code == 400
    assert "Unknown scenario family" in response.json()["detail"]


def test_stress_test_rejects_unsupported_assets():
    latest = client.get("/api/v1/scenarios/latest?event_type=market_crash")
    assert latest.status_code == 200
    scenario_id = latest.json()["id"]

    response = client.post(
        "/api/v1/stress-test/run",
        json={
            "scenario_id": scenario_id,
            "holdings": [
                {"asset": "New Asset", "weight": 100, "amount": 1_000_000, "category": "equity"},
            ],
        },
    )
    assert response.status_code == 400
    assert "Unsupported holdings" in response.json()["detail"]


def test_stress_test_persists_scenario_link():
    latest = client.get("/api/v1/scenarios/latest?event_type=market_crash")
    assert latest.status_code == 200
    scenario_id = latest.json()["id"]

    response = client.post(
        "/api/v1/stress-test/run",
        json={
            "scenario_id": scenario_id,
            "holdings": [
                {"asset": "S&P 500", "weight": 60, "amount": 600_000, "category": "equity"},
                {"asset": "Gold", "weight": 40, "amount": 400_000, "category": "commodities"},
            ],
        },
    )
    assert response.status_code == 200
    result_id = response.json()["id"]

    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT scenario_id FROM app.stress_test_results WHERE id = %s",
        (result_id,),
    )
    stored_scenario_id = cursor.fetchone()[0]
    cursor.execute("DELETE FROM app.stress_test_results WHERE id = %s", (result_id,))
    conn.commit()
    cursor.close()
    conn.close()

    assert str(stored_scenario_id) == scenario_id


def test_top_causal_links_do_not_fabricate_confidence():
    response = client.get("/api/v1/dashboard/top-causal-links?limit=5")
    assert response.status_code == 200
    links = response.json()
    assert links
    assert all(link["confidence"] is None or isinstance(link["confidence"], (int, float)) for link in links)
