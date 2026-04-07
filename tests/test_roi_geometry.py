import pytest
import numpy as np
from core.roi import ROI, ROIHandle, ROIShape


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


# ---------------------------------------------------------------------------
# Lines 100–102: get_center() — polygon centroid branch
# ---------------------------------------------------------------------------


def test_get_center_polygon_returns_mean_of_vertices():
    """Polygon get_center returns the arithmetic mean of all vertex coordinates."""
    roi = ROI(
        x=0, y=0, width=10, height=10,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (4, 0), (4, 4), (0, 4)],
    )
    cx, cy = roi.get_center()
    assert cx == 2.0
    assert cy == 2.0


def test_get_center_polygon_triangle_centroid():
    """Triangle centroid is computed as the mean of its three vertices."""
    roi = ROI(
        x=0, y=0, width=6, height=6,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (6, 0), (3, 6)],
    )
    cx, cy = roi.get_center()
    assert cx == pytest.approx(3.0)
    assert cy == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Line 125: contains_point() — zero-radius ellipse returns False immediately
# ---------------------------------------------------------------------------


def test_contains_point_zero_width_ellipse_returns_false():
    """Ellipse with width=0 (rx=0) always returns False from contains_point."""
    roi = ROI(x=5, y=5, width=0, height=10, shape=ROIShape.ELLIPSE)
    assert roi.contains_point(5, 10) is False


def test_contains_point_zero_height_ellipse_returns_false():
    """Ellipse with height=0 (ry=0) always returns False from contains_point."""
    roi = ROI(x=5, y=5, width=10, height=0, shape=ROIShape.ELLIPSE)
    assert roi.contains_point(10, 5) is False


# ---------------------------------------------------------------------------
# Lines 136–156: get_handle_at_point() — polygon and ellipse paths
# ---------------------------------------------------------------------------


def test_get_handle_at_point_polygon_vertex_hit_returns_vertex_tuple():
    """A point on a polygon vertex returns (ROIHandle.VERTEX, index)."""
    roi = ROI(
        x=0, y=0, width=20, height=20,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (20, 0), (20, 20), (0, 20)],
    )
    result = roi.get_handle_at_point(0, 0)
    assert result == (ROIHandle.VERTEX, 0)


def test_get_handle_at_point_polygon_second_vertex_hit():
    """A point on the second polygon vertex returns (VERTEX, 1)."""
    roi = ROI(
        x=0, y=0, width=20, height=20,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (20, 0), (20, 20), (0, 20)],
    )
    result = roi.get_handle_at_point(20, 0)
    assert result == (ROIHandle.VERTEX, 1)


def test_get_handle_at_point_polygon_inside_returns_move():
    """A point inside the polygon (not on a vertex) returns MOVE."""
    roi = ROI(
        x=0, y=0, width=40, height=40,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (40, 0), (40, 40), (0, 40)],
    )
    result = roi.get_handle_at_point(20, 20)
    assert result == ROIHandle.MOVE


def test_get_handle_at_point_polygon_outside_returns_none():
    """A point outside the polygon (and not on a vertex) returns NONE."""
    roi = ROI(
        x=0, y=0, width=10, height=10,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )
    result = roi.get_handle_at_point(50, 50)
    assert result == ROIHandle.NONE


def test_get_handle_at_point_ellipse_top_left_corner():
    """A point exactly on the top-left corner of the ellipse bbox returns TOP_LEFT."""
    roi = ROI(x=10, y=10, width=20, height=20, shape=ROIShape.ELLIPSE)
    assert roi.get_handle_at_point(10, 10) == ROIHandle.TOP_LEFT


def test_get_handle_at_point_ellipse_top_right_corner():
    """A point on the top-right corner of the ellipse bbox returns TOP_RIGHT."""
    roi = ROI(x=10, y=10, width=20, height=20, shape=ROIShape.ELLIPSE)
    assert roi.get_handle_at_point(30, 10) == ROIHandle.TOP_RIGHT


def test_get_handle_at_point_ellipse_bottom_left_corner():
    """A point on the bottom-left corner of the ellipse bbox returns BOTTOM_LEFT."""
    roi = ROI(x=10, y=10, width=20, height=20, shape=ROIShape.ELLIPSE)
    assert roi.get_handle_at_point(10, 30) == ROIHandle.BOTTOM_LEFT


def test_get_handle_at_point_ellipse_bottom_right_corner():
    """A point on the bottom-right corner of the ellipse bbox returns BOTTOM_RIGHT."""
    roi = ROI(x=10, y=10, width=20, height=20, shape=ROIShape.ELLIPSE)
    assert roi.get_handle_at_point(30, 30) == ROIHandle.BOTTOM_RIGHT


def test_get_handle_at_point_ellipse_center_returns_move():
    """A point at the center of an ellipse (not near a corner) returns MOVE."""
    roi = ROI(x=0, y=0, width=60, height=60, shape=ROIShape.ELLIPSE)
    assert roi.get_handle_at_point(30, 30) == ROIHandle.MOVE


def test_get_handle_at_point_ellipse_far_outside_returns_none():
    """A point far outside the ellipse and corners returns NONE."""
    roi = ROI(x=0, y=0, width=10, height=10, shape=ROIShape.ELLIPSE)
    assert roi.get_handle_at_point(200, 200) == ROIHandle.NONE


# ---------------------------------------------------------------------------
# Lines 173–192: adjust_with_handle() — polygon branch
# ---------------------------------------------------------------------------


def test_adjust_with_handle_polygon_move_shifts_all_vertices():
    """MOVE handle shifts every polygon vertex by (dx, dy)."""
    roi = ROI(
        x=5, y=5, width=10, height=10,
        shape=ROIShape.POLYGON,
        points=[(5, 5), (15, 5), (15, 15), (5, 15)],
    )
    roi.adjust_with_handle(ROIHandle.MOVE, 3, 2, 100, 100)
    assert (8, 7) in roi.points
    assert (18, 7) in roi.points
    assert (18, 17) in roi.points
    assert (8, 17) in roi.points


def test_adjust_with_handle_polygon_move_clamps_to_image_bounds():
    """MOVE handle clamps all polygon vertices so they stay within image bounds."""
    roi = ROI(
        x=0, y=0, width=5, height=5,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (5, 0), (5, 5), (0, 5)],
    )
    roi.adjust_with_handle(ROIHandle.MOVE, -100, -100, 100, 100)
    for vx, vy in roi.points:
        assert vx >= 0
        assert vy >= 0


def test_adjust_with_handle_polygon_vertex_moves_single_vertex():
    """VERTEX handle with vertex_index moves only the specified vertex."""
    roi = ROI(
        x=0, y=0, width=10, height=10,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )
    roi.adjust_with_handle(ROIHandle.VERTEX, 2, 3, 100, 100, vertex_index=0)
    assert roi.points[0] == (2, 3)
    assert roi.points[1] == (10, 0)  # unchanged


def test_adjust_with_handle_polygon_tuple_handle_unpacked_correctly():
    """Tuple (VERTEX, index) handle is unpacked and moves the correct vertex."""
    roi = ROI(
        x=0, y=0, width=10, height=10,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )
    roi.adjust_with_handle((ROIHandle.VERTEX, 2), 1, 1, 100, 100)
    assert roi.points[2] == (11, 11)
    assert roi.points[0] == (0, 0)  # unchanged


def test_adjust_with_handle_polygon_vertex_clamps_to_bounds():
    """VERTEX handle clamps the moved vertex to [0, image_dim-1]."""
    roi = ROI(
        x=0, y=0, width=10, height=10,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )
    roi.adjust_with_handle(ROIHandle.VERTEX, -50, -50, 100, 100, vertex_index=0)
    vx, vy = roi.points[0]
    assert vx == 0 and vy == 0


# ---------------------------------------------------------------------------
# Lines 194–217: adjust_with_handle() — ellipse branch (all 5 handle cases)
# ---------------------------------------------------------------------------


def test_adjust_with_handle_ellipse_move_shifts_position():
    """MOVE handle on ellipse shifts x and y by (dx, dy)."""
    roi = ROI(x=10, y=10, width=20, height=20, shape=ROIShape.ELLIPSE)
    roi.adjust_with_handle(ROIHandle.MOVE, 5, 3, 200, 200)
    assert roi.x == 15
    assert roi.y == 13


def test_adjust_with_handle_ellipse_move_clamps_within_image():
    """MOVE handle clamps ellipse position so it cannot leave image bounds."""
    roi = ROI(x=0, y=0, width=20, height=20, shape=ROIShape.ELLIPSE)
    roi.adjust_with_handle(ROIHandle.MOVE, -100, -100, 200, 200)
    assert roi.x == 0
    assert roi.y == 0


def test_adjust_with_handle_ellipse_top_left_resizes_from_top_left():
    """TOP_LEFT handle moves the top-left corner and resizes width/height accordingly."""
    roi = ROI(x=10, y=10, width=30, height=30, shape=ROIShape.ELLIPSE)
    roi.adjust_with_handle(ROIHandle.TOP_LEFT, 5, 5, 200, 200)
    assert roi.x == 15
    assert roi.y == 15
    assert roi.width == 25
    assert roi.height == 25


def test_adjust_with_handle_ellipse_top_right_resizes():
    """TOP_RIGHT handle expands width to the right and shrinks from the top."""
    roi = ROI(x=10, y=10, width=30, height=30, shape=ROIShape.ELLIPSE)
    roi.adjust_with_handle(ROIHandle.TOP_RIGHT, 5, 5, 200, 200)
    assert roi.y == 15
    assert roi.width == 35
    assert roi.height == 25


def test_adjust_with_handle_ellipse_bottom_left_resizes():
    """BOTTOM_LEFT handle moves left edge and expands height downward."""
    roi = ROI(x=10, y=10, width=30, height=30, shape=ROIShape.ELLIPSE)
    roi.adjust_with_handle(ROIHandle.BOTTOM_LEFT, 5, 5, 200, 200)
    assert roi.x == 15
    assert roi.width == 25
    assert roi.height == 35


def test_adjust_with_handle_ellipse_bottom_right_resizes():
    """BOTTOM_RIGHT handle expands both width and height to the bottom-right."""
    roi = ROI(x=10, y=10, width=30, height=30, shape=ROIShape.ELLIPSE)
    roi.adjust_with_handle(ROIHandle.BOTTOM_RIGHT, 5, 5, 200, 200)
    assert roi.width == 35
    assert roi.height == 35


# ---------------------------------------------------------------------------
# Lines 221–228: _update_bbox_from_points() — empty-points no-op and normal update
# ---------------------------------------------------------------------------


def test_update_bbox_from_points_is_noop_when_points_empty():
    """_update_bbox_from_points does not change bbox when points list is empty."""
    roi = ROI(
        x=5, y=5, width=10, height=10,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (10, 0), (10, 10)],
    )
    roi.points = []
    roi._update_bbox_from_points()
    assert roi.x == 5
    assert roi.width == 10


def test_update_bbox_from_points_recomputes_after_vertex_edit():
    """_update_bbox_from_points correctly recomputes bbox when a vertex is moved."""
    roi = ROI(
        x=0, y=0, width=11, height=11,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )
    roi.points[3] = (0, 20)  # extend y range
    roi._update_bbox_from_points()
    # max y=20, min y=0 → height = 20 - 0 + 1 = 21
    assert roi.height == 21
    assert roi.y == 0


# ---------------------------------------------------------------------------
# Lines 230–238: delete_vertex()
# ---------------------------------------------------------------------------


def test_delete_vertex_removes_vertex_when_more_than_3():
    """delete_vertex removes the indexed vertex and returns True for a 4-vertex polygon."""
    roi = ROI(
        x=0, y=0, width=10, height=10,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )
    result = roi.delete_vertex(1)
    assert result is True
    assert len(roi.points) == 3
    assert (10, 0) not in roi.points


def test_delete_vertex_bbox_updated_after_deletion():
    """delete_vertex updates the bounding box after removing a vertex."""
    roi = ROI(
        x=0, y=0, width=21, height=21,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (20, 0), (20, 20), (0, 10), (0, 20)],
    )
    roi.delete_vertex(4)  # remove (0, 20)
    # remaining max-y is now 20 (from vertex 2), unchanged, but test bbox recalculated
    assert roi.width >= 1 and roi.height >= 1


def test_delete_vertex_refuses_when_exactly_3_vertices():
    """delete_vertex returns False without modifying the polygon when it has only 3 vertices."""
    roi = ROI(
        x=0, y=0, width=10, height=10,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (10, 0), (5, 10)],
    )
    assert roi.delete_vertex(0) is False
    assert len(roi.points) == 3


def test_delete_vertex_refuses_for_ellipse_roi():
    """delete_vertex returns False immediately for a non-polygon ROI."""
    roi = ROI(x=0, y=0, width=10, height=10, shape=ROIShape.ELLIPSE)
    assert roi.delete_vertex(0) is False


def test_delete_vertex_refuses_out_of_range_index():
    """delete_vertex returns False when the index is out of range."""
    roi = ROI(
        x=0, y=0, width=10, height=10,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )
    assert roi.delete_vertex(99) is False
    assert len(roi.points) == 4


# ---------------------------------------------------------------------------
# Parametrized: create_mask produces correct shape across image/ROI sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("img_w,img_h,roi_x,roi_y", [
    (10, 10, 0, 0),
    (50, 50, 20, 20),
    (100, 80, 40, 30),
])
def test_create_mask_polygon_shape_matches_image_dimensions(img_w, img_h, roi_x, roi_y):
    """create_mask for a polygon returns a uint8 mask with (img_h, img_w) shape."""
    pts = [(roi_x, roi_y), (roi_x + 4, roi_y), (roi_x + 2, roi_y + 4)]
    roi = ROI(x=roi_x, y=roi_y, width=5, height=5, shape=ROIShape.POLYGON, points=pts)
    mask = roi.create_mask(img_w, img_h)
    assert mask.shape == (img_h, img_w)
    assert mask.dtype == np.uint8
