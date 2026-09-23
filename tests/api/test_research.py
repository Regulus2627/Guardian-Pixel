"""Tests for the real GuardianPixel research-results API."""

from __future__ import annotations

from pathlib import Path

from backend.app import create_app


REAL_RESULT_DIRECTORY = Path(
    "research_assets/steganalysis"
)


def test_research_endpoint_serves_real_held_out_results() -> None:
    app = create_app(
        {
            "TESTING": True,
            "STEGANALYSIS_DIRECTORY": str(REAL_RESULT_DIRECTORY),
        }
    )
    response = app.test_client().get("/api/v1/research")

    assert response.status_code == 200
    data = response.get_json()

    assert data["isSimulated"] is False
    assert data["preliminary"] is False
    assert data["dataset"]["coverCount"] == 20
    assert data["dataset"]["resultRows"] == 300
    assert data["dataset"]["methodCount"] == 5
    assert data["dataset"]["payloadLevelsBytes"] == [500, 2000, 5000]
    assert data["dataset"]["equalBitComparison"] is True
    assert data["dataset"]["equalBppWithinEachCondition"] is True
    assert data["batchErrors"] == []
    assert len(data["runs"]) == 300

    methods = {
        row["method"]: row
        for row in data["methodSummary"]
    }

    assert methods["GuardianPixel"]["totalRuns"] == 60
    assert methods["GuardianPixel"]["reliableRuns"] == 60
    assert methods["GuardianPixel"]["reliableSuccessRatePercent"] == 100.0
    assert methods["GuardianPixel"]["reliableMetrics"]["histogramL1"] is not None
    assert methods["GuardianPixel"]["reliableMetrics"]["suitabilityGain"] is not None

    assert methods["Sequential Replacement"]["reliableRuns"] == 60
    assert methods["Sequential Matching"]["reliableRuns"] == 60
    assert methods["Keyed Random Matching"]["reliableRuns"] == 60

    assert methods["Canny Synchronous"]["reliableRuns"] == 0
    assert methods["Canny Synchronous"]["includedInReliableRanking"] is False
    assert data["cannyDiagnostics"]["reliableRuns"] == 0


def test_research_endpoint_reports_missing_result_directory(
    tmp_path: Path,
) -> None:
    missing_directory = tmp_path / "missing-results"

    app = create_app(
        {
            "TESTING": True,
            "STEGANALYSIS_DIRECTORY": str(missing_directory),
        }
    )
    response = app.test_client().get("/api/v1/research")

    assert response.status_code == 404
    data = response.get_json()
    assert data["code"] == "RESEARCH_RESULT_NOT_FOUND"


def test_research_endpoint_rejects_invalid_methodology_json(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "invalid-results"
    directory.mkdir()

    valid_csv = "method,image_id,requested_payload_bytes,cover_category\n"
    valid_csv += "GuardianPixel,test.png,500,Mixed\n"

    (directory / "steganalysis_runs.csv").write_text(
        valid_csv,
        encoding="utf-8",
    )
    (directory / "steganalysis_method_summary.csv").write_text(
        "method,total_runs\nGuardianPixel,1\n",
        encoding="utf-8",
    )
    (directory / "steganalysis_category_summary.csv").write_text(
        "cover_category,method,total_runs\nMixed,GuardianPixel,1\n",
        encoding="utf-8",
    )
    (directory / "methodology.json").write_text(
        "not valid JSON",
        encoding="utf-8",
    )
    (directory / "batch_errors.json").write_text(
        "[]",
        encoding="utf-8",
    )

    app = create_app(
        {
            "TESTING": True,
            "STEGANALYSIS_DIRECTORY": str(directory),
        }
    )
    response = app.test_client().get("/api/v1/research")

    assert response.status_code == 500
    assert response.get_json()["code"] == "INVALID_RESEARCH_RESULT"
