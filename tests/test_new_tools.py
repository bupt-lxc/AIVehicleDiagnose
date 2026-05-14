import pytest
from app.diagnostic.tools.dtc_parser import extract_dtc_codes, validate_dtc_format
from app.diagnostic.tools.freeze_frame_parser import parse_freeze_frame, filter_relevant_parameters

def test_extract_dtc_codes():
    text = "Error codes: P0123, C0045 and B1234 found."
    codes = extract_dtc_codes(text)
    assert codes == ["B1234", "C0045", "P0123"]
    assert extract_dtc_codes("") == []
    assert extract_dtc_codes("No codes here") == []

def test_validate_dtc_format():
    assert validate_dtc_format("P0123") is True
    assert validate_dtc_format("p0123") is True
    assert validate_dtc_format("1234") is False
    assert validate_dtc_format("P12345") is False

def test_parse_freeze_frame():
    text = """
    RPM = 2000
    Speed: 60 km/h
    Load= 50%
    """
    data = parse_freeze_frame(text)
    assert data["RPM"] == "2000"
    assert data["Speed"] == "60 km/h"
    assert data["Load"] == "50%"

def test_filter_relevant_parameters():
    data = {"RPM": "2000", "Speed": "60", "Temp": "90"}
    filtered = filter_relevant_parameters(data, ["rpm", "temp"])
    assert "RPM" in filtered
    assert "Temp" in filtered
    assert "Speed" not in filtered
