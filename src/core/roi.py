"""ROI (Region of Interest) data structures and utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


class ROIShape(Enum):
    """ROI shape types."""
    ELLIPSE = "ellipse"  # Legacy support only, not used in UI
    POLYGON = "polygon"


class ROIHandle(Enum):
    """ROI adjustment handles."""
    NONE = 0
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4
    MOVE = 5  # For dragging the entire ROI
    VERTEX = 6  # Polygon vertex (use with vertex_index)


HandleResult = Union[ROIHandle, Tuple[ROIHandle, int]]


@dataclass
class ROI:
    """
    Region of Interest (ellipse or polygon).
    
    Coordinates are stored in original image pixel space, not display space.
    This ensures ROI stays fixed to image region regardless of scaling.
    
    Structure:
    - For ELLIPSE: x, y, width, height define bounding box; center and radii derived.
    - For POLYGON: points list defines vertices; x, y, width, height are bounding box.
    """
    x: int
    y: int
    width: int
    height: int
    shape: ROIShape = ROIShape.ELLIPSE
    points: Optional[List[Tuple[int, int]]] = field(default=None)
    
    def to_dict(self) -> dict:
        """Convert ROI to dictionary for serialization."""
        d = {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "shape": self.shape.value
        }
        if self.shape == ROIShape.POLYGON and self.points is not None:
            d["points"] = [[int(p[0]), int(p[1])] for p in self.points]
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> ROI:
        """Create ROI from dictionary with safe defaults for missing keys."""
        shape_str = data.get("shape", "ellipse")
        try:
            shape = ROIShape(shape_str)
        except ValueError:
            shape = ROIShape.ELLIPSE
        
        if shape == ROIShape.POLYGON and "points" in data:
            pts = data["points"]
            if isinstance(pts, (list, tuple)) and len(pts) >= 3:
                points = [(int(p[0]), int(p[1])) for p in pts]
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x = min(x_coords)
                y = min(y_coords)
                width = max(x_coords) - x + 1
                height = max(y_coords) - y + 1
                return cls(x=x, y=y, width=width, height=height, shape=shape, points=points)
        
        return cls(
            x=data.get("x", 0),
            y=data.get("y", 0),
            width=data.get("width", 100),
            height=data.get("height", 100),
            shape=shape
        )
    
    def get_center(self) -> Tuple[float, float]:
        """Get center coordinates of ROI."""
        if self.shape == ROIShape.POLYGON and self.points and len(self.points) > 0:
            cx = sum(p[0] for p in self.points) / len(self.points)
            cy = sum(p[1] for p in self.points) / len(self.points)
            return (cx, cy)
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Return (x, y, width, height) bounding box."""
        if self.shape == ROIShape.POLYGON and self.points and len(self.points) > 0:
            x_coords = [p[0] for p in self.points]
            y_coords = [p[1] for p in self.points]
            x = min(x_coords)
            y = min(y_coords)
            return (x, y, max(x_coords) - x + 1, max(y_coords) - y + 1)
        return (self.x, self.y, self.width, self.height)
    
    def contains_point(self, px: int, py: int) -> bool:
        """Check if a point is inside the ROI."""
        if self.shape == ROIShape.POLYGON and self.points and len(self.points) >= 3:
            contour = np.array(self.points, dtype=np.int32)
            return cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0
        # Ellipse
        cx, cy = self.get_center()
        rx = self.width / 2
        ry = self.height / 2
        if rx == 0 or ry == 0:
            return False
        normalized = ((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2
        return normalized <= 1.0
    
    def get_handle_at_point(self, px: int, py: int, handle_size: int = 8) -> HandleResult:
        """
        Determine which handle (if any) is at the given point.
        
        Returns:
            ROIHandle, or (ROIHandle.VERTEX, index) for polygon vertex
        """
        if self.shape == ROIShape.POLYGON and self.points and len(self.points) > 0:
            for i, (vx, vy) in enumerate(self.points):
                if abs(px - vx) <= handle_size and abs(py - vy) <= handle_size:
                    return (ROIHandle.VERTEX, i)
            if self.contains_point(px, py):
                return ROIHandle.MOVE
            return ROIHandle.NONE
        
        # Ellipse
        corners = [
            (self.x, self.y, ROIHandle.TOP_LEFT),
            (self.x + self.width, self.y, ROIHandle.TOP_RIGHT),
            (self.x, self.y + self.height, ROIHandle.BOTTOM_LEFT),
            (self.x + self.width, self.y + self.height, ROIHandle.BOTTOM_RIGHT),
        ]
        for cx, cy, handle in corners:
            if abs(px - cx) <= handle_size and abs(py - cy) <= handle_size:
                return handle
        if self.contains_point(px, py):
            return ROIHandle.MOVE
        return ROIHandle.NONE
    
    def adjust_with_handle(
        self, 
        handle: Union[ROIHandle, HandleResult], 
        dx: int, 
        dy: int,
        image_width: int,
        image_height: int,
        vertex_index: Optional[int] = None
    ) -> None:
        """
        Adjust ROI based on handle being dragged.
        
        For polygon: pass (ROIHandle.VERTEX, index) as handle, or handle=ROIHandle.VERTEX
        with vertex_index set.
        """
        vidx = vertex_index
        if isinstance(handle, tuple):
            handle, vidx = handle
        
        if self.shape == ROIShape.POLYGON and self.points is not None:
            if handle == ROIHandle.MOVE:
                new_points = []
                for vx, vy in self.points:
                    nx = max(0, min(image_width - 1, vx + dx))
                    ny = max(0, min(image_height - 1, vy + dy))
                    new_points.append((nx, ny))
                self.points = new_points
                self._update_bbox_from_points()
            elif handle == ROIHandle.VERTEX and vidx is not None and 0 <= vidx < len(self.points):
                vx, vy = self.points[vidx]
                nx = max(0, min(image_width - 1, vx + dx))
                ny = max(0, min(image_height - 1, vy + dy))
                self.points[vidx] = (nx, ny)
                self._update_bbox_from_points()
            return
        
        # Ellipse
        if handle == ROIHandle.MOVE:
            self.x = max(0, min(image_width - self.width, self.x + dx))
            self.y = max(0, min(image_height - self.height, self.y + dy))
        elif handle == ROIHandle.TOP_LEFT:
            new_x = max(0, min(self.x + self.width - 10, self.x + dx))
            new_y = max(0, min(self.y + self.height - 10, self.y + dy))
            self.width += self.x - new_x
            self.height += self.y - new_y
            self.x = new_x
            self.y = new_y
        elif handle == ROIHandle.TOP_RIGHT:
            new_y = max(0, min(self.y + self.height - 10, self.y + dy))
            self.width = max(10, min(image_width - self.x, self.width + dx))
            self.height += self.y - new_y
            self.y = new_y
        elif handle == ROIHandle.BOTTOM_LEFT:
            new_x = max(0, min(self.x + self.width - 10, self.x + dx))
            self.width += self.x - new_x
            self.height = max(10, min(image_height - self.y, self.height + dy))
            self.x = new_x
        elif handle == ROIHandle.BOTTOM_RIGHT:
            self.width = max(10, min(image_width - self.x, self.width + dx))
            self.height = max(10, min(image_height - self.y, self.height + dy))
    
    def _update_bbox_from_points(self) -> None:
        """Update x, y, width, height from points (for polygon)."""
        if not self.points or len(self.points) == 0:
            return
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        self.x = min(x_coords)
        self.y = min(y_coords)
        self.width = max(x_coords) - self.x + 1
        self.height = max(y_coords) - self.y + 1
    
    def delete_vertex(self, index: int) -> bool:
        """Delete polygon vertex. Returns True if deleted (only if > 3 vertices)."""
        if self.shape != ROIShape.POLYGON or not self.points or len(self.points) <= 3:
            return False
        if 0 <= index < len(self.points):
            self.points.pop(index)
            self._update_bbox_from_points()
            return True
        return False
    
    def create_mask(self, image_width: int, image_height: int) -> np.ndarray:
        """
        Create a binary mask for the ROI.
        
        Returns:
            Binary mask (uint8) where ROI region is 255, rest is 0
        """
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        if self.shape == ROIShape.POLYGON and self.points and len(self.points) >= 3:
            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            return mask
        
        # Ellipse
        cx, cy = self.get_center()
        rx = self.width / 2
        ry = self.height / 2
        y_coords, x_coords = np.ogrid[:image_height, :image_width]
        if rx > 0 and ry > 0:
            ellipse_mask = ((x_coords - cx) / rx) ** 2 + ((y_coords - cy) / ry) ** 2 <= 1
            mask[ellipse_mask] = 255
        return mask
