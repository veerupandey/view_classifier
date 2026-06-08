from view_classifier.data import build_manifest, split_identities
from view_classifier.labels import angle_to_pose_idx


def test_split_covers_all_identities_without_overlap():
    ids = list(range(47))
    splits = split_identities(ids, seed=42)
    all_ids = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_ids) == ids
    assert not set(splits["train"]) & set(splits["val"])
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["val"]) & set(splits["test"])
    assert splits["train"] and splits["val"] and splits["test"]


def test_build_manifest_tags_source_and_bins():
    rows = [
        {"identity": 1, "angle": 0.0, "x1": "41.9", "y1": "63.4", "x2": "880.0", "y2": "412.5", "source_path": "car001/a.png"},
        {"identity": 2, "angle": 90.0, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "source_path": "car002/a.png"},
        {"identity": 3, "angle": 180.0, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "source_path": "car003/a.png"},
    ]
    m = build_manifest(rows, seed=0, dataset="unsupcar", source="unsupcar")
    assert m["source"] == "unsupcar"
    assert m["n_identities"] == 3
    assert {r["source"] for r in m["records"]} == {"unsupcar"}
    by_id = {r["identity"]: r for r in m["records"]}
    assert by_id[1]["bbox"] == [41, 63, 880, 412]
    assert by_id[1]["pose_idx"] == angle_to_pose_idx(0)
    assert by_id[2]["pose"] == "right"
    assert by_id[3]["source_path"] == "car003/a.png"
