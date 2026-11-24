import pytest

from mlflowops import data
from mlflowops.errors import ServiceError
from mlflowops.sample_data import generate


def csv_bytes(df):
    return df.to_csv(index=False).encode()


def test_parse_ok_and_profile():
    df = data.parse_csv(csv_bytes(generate(100)), "a.csv", 5)
    p = data.profile(df)
    assert p["rows"] == 100 and p["cols"] == 8
    assert any(c["name"] == "total_charges" and c["missing"] > 0 for c in p["columns"])


@pytest.mark.parametrize("content,name", [(b"", "a.csv"), (b"a,b\n1,2\n", "a.csv"), (b"x", "a.txt")])
def test_parse_rejects_bad_input(content, name):
    with pytest.raises(ServiceError):
        data.parse_csv(content, name, 5)


def test_size_limit():
    with pytest.raises(ServiceError) as e:
        data.parse_csv(csv_bytes(generate(2000)), "a.csv", 0)
    assert e.value.status == 413


def test_target_checks():
    df = generate(100)
    data.check_target(df, "churn")
    with pytest.raises(ServiceError):
        data.check_target(df, "nope")
    df["const"] = 1
    with pytest.raises(ServiceError):
        data.check_target(df, "const")
