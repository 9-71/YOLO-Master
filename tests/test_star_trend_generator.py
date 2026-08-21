from datetime import datetime, timezone

from scripts.generate_star_trend import daily_series, render_svg


def test_daily_series_fills_missing_days_and_accumulates():
    timestamps = [
        datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 3, 2, tzinfo=timezone.utc),
        datetime(2026, 1, 3, 3, tzinfo=timezone.utc),
    ]

    series = daily_series(timestamps)

    assert [(item[0].date().isoformat(), item[1], item[2]) for item in series] == [
        ("2026-01-01", 1, 1),
        ("2026-01-02", 0, 1),
        ("2026-01-03", 2, 3),
    ]


def test_render_svg_contains_accessible_metadata_and_series():
    timestamps = [
        datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 2, 2, tzinfo=timezone.utc),
    ]
    series = daily_series(timestamps)

    svg = render_svg("Tencent/YOLO-Master", series, timestamps[-1])

    assert 'role="img"' in svg
    assert "Cumulative GitHub stars from 2026-01-01 to 2026-01-02; 2 stars" in svg
    assert '<polyline class="line"' in svg
    assert '<g class="bar">' in svg
