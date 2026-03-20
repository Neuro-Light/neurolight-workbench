from core.roi import ROI, ROIShape


def test_roi_polygon_roundtrip_preserves_points_and_bounds():
    points = [(2, 3), (6, 3), (5, 7), (2, 8)]
    roi = ROI(x=2, y=3, width=5, height=5, shape=ROIShape.POLYGON, points=points)

    data = roi.to_dict()
    restored = ROI.from_dict(data)

    assert data["shape"] == "polygon"
    assert "points" in data and data["points"] == [[p[0], p[1]] for p in points]
    assert restored.shape == ROIShape.POLYGON
    assert restored.points == points
    assert restored.get_bounding_box() == (2, 3, 5, 6)


def test_roi_from_dict_with_invalid_shape_falls_back_to_ellipse():
    restored = ROI.from_dict({"shape": "triangle", "x": 10, "y": 12, "width": 30, "height": 40})

    assert restored.shape == ROIShape.ELLIPSE
    assert restored.get_bounding_box() == (10, 12, 30, 40)


def test_roi_contains_point_and_create_mask_work_for_both_shapes():
    ellipse = ROI(x=5, y=5, width=10, height=6, shape=ROIShape.ELLIPSE)
    polygon = ROI(
        x=2,
        y=2,
        width=4,
        height=4,
        shape=ROIShape.POLYGON,
        points=[(2, 2), (6, 2), (6, 6), (2, 6)],
    )

    assert ellipse.contains_point(10, 8) is True
    assert ellipse.contains_point(0, 0) is False
    assert polygon.contains_point(3, 3) is True
    assert polygon.contains_point(0, 0) is False

    ellipse_mask = ellipse.create_mask(20, 20)
    polygon_mask = polygon.create_mask(20, 20)

    assert ellipse_mask[8, 10] == 255
    assert ellipse_mask[0, 0] == 0
    assert polygon_mask[3, 3] == 255
    assert polygon_mask[0, 0] == 0
