import pandas as pd
import pytest

from pqr_sentiment.config import RAW_COLUMNS
from pqr_sentiment.data import (
    DataValidationError,
    clean_records,
    load_records,
    parse_flexible_dates,
)


def make_raw_frame() -> pd.DataFrame:
    """Synthetic rows reproducing every defect observed in the source system."""
    rows = [
        # 0: fully clean complaint
        {
            "record_id": 1,
            "pqr_type": "Queja",
            "received_date": "2017-10-20 00:00:00",
            "due_date": "2017-10-25 00:00:00",
            "attended_date": "2017-10-20 00:00:00",
            "closed_date": "2017-10-24 00:00:00",
            "entry_route": "Contratista",
            "watershed": "Cuenca Alta",
            "municipality": "Municipio Uno",
            "neighborhood": "Barrio Centro",
            "description": "Se presenta filtracion de agua en la via",
            "attributable": "Imputable",
            "status": "Cerrada",
        },
        # 1: trailing space in category, Excel serial date, case variant attributable
        {
            "record_id": 2,
            "pqr_type": "Queja ",
            "received_date": 43028,
            "due_date": 43033,
            "attended_date": 43028,
            "closed_date": 43038,
            "entry_route": "Interventoría",
            "watershed": "Cuenca Alta",
            "municipality": "Municipio Uno",
            "neighborhood": "Barrio centro",
            "description": "Dano en anden por maquinaria pesada",
            "attributable": "no Imputable",
            "status": "Cerrada",
        },
        # 2: petition through the utility, day first date string, shared fault
        {
            "record_id": 3,
            "pqr_type": "Petición",
            "received_date": "24/10/2017",
            "due_date": "29/10/2017",
            "attended_date": "24/10/2017",
            "closed_date": "05/11/2017",
            "entry_route": "Empresa de servicios",
            "watershed": "Cuenca Baja",
            "municipality": "Municipio Dos",
            "neighborhood": "Barrio Norte",
            "description": "Solicita informacion de obras en el sector",
            "attributable": "Compartida",
            "status": "Abierta",
        },
        # 3: malformed date, must be dropped with a recorded reason
        {
            "record_id": 4,
            "pqr_type": "Reclamo",
            "received_date": "06/05/201",
            "due_date": "11/05/2019",
            "attended_date": "06/05/2019",
            "closed_date": "20/05/2019",
            "entry_route": "Contratista",
            "watershed": "Cuenca Baja",
            "municipality": "Municipio Dos",
            "neighborhood": "Barrio Sur",
            "description": "Reclama por dano en fachada",
            "attributable": "NO Imputable",
            "status": "Cerrada",
        },
        # 4: missing essential description, must be dropped
        {
            "record_id": 5,
            "pqr_type": "Queja",
            "received_date": "2018-02-01 00:00:00",
            "due_date": "2018-02-06 00:00:00",
            "attended_date": "2018-02-01 00:00:00",
            "closed_date": "2018-02-03 00:00:00",
            "entry_route": "Contratista",
            "watershed": "Cuenca Alta",
            "municipality": "Municipio Uno",
            "neighborhood": "Barrio Centro",
            "description": None,
            "attributable": "Imputable",
            "status": "Cerrada",
        },
        # 5: open record without closing date, resolution stays null
        {
            "record_id": 6,
            "pqr_type": "Reclamo",
            "received_date": "2018-03-10 00:00:00",
            "due_date": "2018-03-15 00:00:00",
            "attended_date": "2018-03-10 00:00:00",
            "closed_date": None,
            "entry_route": "Interventoría",
            "watershed": "Cuenca Media",
            "municipality": "Municipio Tres",
            "neighborhood": "Barrio Oriente",
            "description": "Reclama compensacion por perdida de ingresos",
            "attributable": "imputable",
            "status": "Abierta",
        },
    ]
    return pd.DataFrame(rows)


class TestLoadRecords:
    def test_load_csv_roundtrip(self, tmp_path):
        path = tmp_path / "records.csv"
        make_raw_frame().to_csv(path, index=False, encoding="utf-8")
        df = load_records(path)
        assert list(df.columns) == RAW_COLUMNS
        assert len(df) == 6

    def test_load_excel_roundtrip(self, tmp_path):
        path = tmp_path / "records.xlsx"
        make_raw_frame().to_excel(path, index=False)
        df = load_records(path)
        assert list(df.columns) == RAW_COLUMNS

    def test_missing_columns_raise(self, tmp_path):
        path = tmp_path / "records.csv"
        make_raw_frame().drop(columns=["watershed"]).to_csv(path, index=False)
        with pytest.raises(DataValidationError, match="watershed"):
            load_records(path)


class TestParseFlexibleDates:
    def test_mixed_formats(self):
        s = pd.Series(["2017-10-20 00:00:00", 43028, "24/10/2017", "06/05/201", None])
        parsed = parse_flexible_dates(s)
        assert parsed[0] == pd.Timestamp("2017-10-20")
        assert parsed[1] == pd.Timestamp("2017-10-20")
        assert parsed[2] == pd.Timestamp("2017-10-24")
        assert pd.isna(parsed[3])
        assert pd.isna(parsed[4])

    def test_serial_strings_also_parse(self):
        parsed = parse_flexible_dates(pd.Series(["43028"]))
        assert parsed[0] == pd.Timestamp("2017-10-20")


class TestCleanRecords:
    @pytest.fixture()
    def cleaned(self):
        return clean_records(make_raw_frame())

    def test_row_accounting(self, cleaned):
        df, report = cleaned
        assert report.rows_in == 6
        assert report.rows_out == 4
        assert len(df) == 4
        assert len(report.dropped) == 2
        reasons = " ".join(report.dropped["drop_reason"].tolist())
        assert "received_date" in reasons
        assert "description" in reasons

    def test_category_normalization(self, cleaned):
        df, _ = cleaned
        assert set(df["pqr_type"]) <= {"petition", "complaint", "claim"}
        assert set(df["entry_route"]) <= {"contractor", "supervision", "utility_company"}
        assert set(df["attributable"]) <= {"attributable", "not_attributable", "shared"}
        assert set(df["status"]) <= {"closed", "open"}

    def test_trailing_space_category_merged(self, cleaned):
        df, report = cleaned
        assert (df["pqr_type"] == "complaint").sum() == 2
        assert report.fixes["whitespace_trimmed_cells"] >= 1

    def test_neighborhood_case_variants_merge_to_most_frequent(self):
        frame = make_raw_frame()
        df, report = clean_records(frame)
        assert (df["neighborhood"] == "Barrio Centro").sum() == 2
        assert "Barrio centro" not in set(df["neighborhood"])
        assert report.fixes["neighborhood_variants_merged"] == 1

    def test_dates_parsed_and_derived_fields(self, cleaned):
        df, _ = cleaned
        row = df.loc[df["record_id"] == 1].iloc[0]
        assert row["received_date"] == pd.Timestamp("2017-10-20")
        assert row["resolution_days"] == 4
        assert bool(row["sla_met"]) is True
        assert row["received_month"] == pd.Timestamp("2017-10-01")

    def test_open_record_keeps_null_resolution(self, cleaned):
        df, _ = cleaned
        row = df.loc[df["record_id"] == 6].iloc[0]
        assert pd.isna(row["resolution_days"])
        assert pd.isna(row["sla_met"])

    def test_late_closure_fails_sla(self, cleaned):
        df, _ = cleaned
        row = df.loc[df["record_id"] == 3].iloc[0]
        assert row["resolution_days"] == 12
        assert bool(row["sla_met"]) is False

    def test_report_frame_renders(self, cleaned):
        _, report = cleaned
        frame = report.to_frame()
        assert {"metric", "value"} == set(frame.columns)
        assert (frame["metric"] == "rows_in").any()
