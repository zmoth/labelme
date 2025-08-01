import collections
from typing import Optional

import imgviz
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

import osam
import numpy as np
from datetime import datetime
from labelme._automation import polygon_from_mask
from labelme._automation import decoder
import labelme.utils
from labelme.shape import Shape

# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = Qt.CursorShape.ArrowCursor
CURSOR_POINT = Qt.CursorShape.PointingHandCursor
CURSOR_DRAW = Qt.CursorShape.CrossCursor
CURSOR_MOVE = Qt.CursorShape.ClosedHandCursor
CURSOR_GRAB = Qt.CursorShape.OpenHandCursor

MOVE_SPEED = 5.0


class Canvas(QtWidgets.QWidget):
    zoomRequest = QtCore.pyqtSignal(int, QtCore.QPoint)
    scrollRequest = QtCore.pyqtSignal(int, int)
    moveRequest = QtCore.pyqtSignal(QtCore.QPointF)
    newShape = QtCore.pyqtSignal(Shape)
    selectionChanged = QtCore.pyqtSignal(list)
    shapeMoved = QtCore.pyqtSignal()
    drawingPolygon = QtCore.pyqtSignal(bool)
    vertexSelected = QtCore.pyqtSignal(bool)
    mouseMoved = QtCore.pyqtSignal(QtCore.QPointF)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(self.double_click)
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        self._crosshair = kwargs.pop(
            "crosshair",
            {
                "polygon": False,
                "rectangle": True,
                "circle": False,
                "line": False,
                "point": False,
                "linestrip": False,
                "ai_polygon": False,
                "ai_mask": False,
                "barcode": False,
                "ai_barcode": False,
            },
        )
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode: int = self.EDIT
        self.shapes: list[Shape] = []
        self.shapesBackups: list[list[Shape]] = []
        self.current: Shape | None = None
        self.selectedShapes: list[Shape] = []  # save the selected shapes here
        self.selectedShapesCopy: list[Shape] = []
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape()
        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.highlightShape: Shape | None = None
        self.prevHighlightShape: Shape | None = None
        self.highlightVertex = None
        self.prevHighlightVertex = None
        self.highlightEdge = None
        self.prevHighlightEdge = None
        self.movingShape = False
        self.snapping = True
        self.highlightShapeIsSelected = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

        self._sam: Optional[osam.types.Model] = None
        self._sam_embedding: collections.OrderedDict[
            bytes, osam.types.ImageEmbedding
        ] = collections.OrderedDict()

        # 画布拖拽
        self._drag_start_position = QtCore.QPoint()  # 记录开始拖动的位置
        self._dragging = False

        # 多选框
        self.select_begin = QtCore.QPoint()
        self.select_end = QtCore.QPoint()
        self._selecting = False

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
            "ai_polygon",
            "ai_mask",
            "barcode",
            "ai_barcode",
        ]:
            raise ValueError("Unsupported createMode: %s" % value)
        self._createMode = value

    def _compute_and_cache_image_embedding(self) -> None:
        if self._sam is None:
            logger.warning("SAM model is not set yet")
            return

        sam: osam.types.Model = self._sam

        image: np.ndarray = labelme.utils.img_qt_to_arr(self.pixmap.toImage())
        if image.tobytes() in self._sam_embedding:
            return

        logger.debug("Computing image embeddings for model {!r}", sam.name)
        self._sam_embedding[image.tobytes()] = sam.encode_image(
            image=imgviz.asrgb(image)
        )

    def _determine_shape_type(self):
        """Return the shape type for the current create mode."""
        if self.createMode in ["ai_polygon", "ai_mask"]:
            return "points"
        elif self.createMode in ["barcode", "ai_barcode"]:
            return "rectangle"  # 将二维码放入矩形框中
        else:
            return self.createMode

    def initializeAiModel(self, model_name):
        if self.pixmap is None:
            logger.warning("Pixmap is not set yet")
            return

        if self._sam is None or self._sam.name != model_name:
            logger.debug("Initializing AI model {!r}", model_name)
            self._sam = osam.apis.get_model_type_by_name(model_name)()
            self._sam_embedding.clear()

        self._compute_and_cache_image_embedding()

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1 :]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev: QtGui.QEnterEvent):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev: QtCore.QEvent):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev: QtGui.QFocusEvent):
        self.restoreCursor()

    def isVisible(self, shape: Shape):  # type: ignore[override]
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value: bool = True):
        self.mode = self.EDIT if value else self.CREATE
        if self.mode == self.EDIT:
            # CREATE -> EDIT
            self.repaint()  # clear crosshair
        else:
            # EDIT -> CREATE
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.highlightShape:
            self.highlightShape.highlightClear()
            self.update()
        self.prevHighlightShape = self.highlightShape
        self.prevHighlightVertex = self.highlightVertex
        self.prevHighlightEdge = self.highlightEdge
        self.highlightShape = self.highlightVertex = self.highlightEdge = None

    def selectedVertex(self):
        return self.highlightVertex is not None

    def selectedEdge(self):
        return self.highlightEdge is not None

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        """Update line with last point and current coordinates."""
        try:
            pos = self.transformPos(ev.localPos())
        except AttributeError:
            return

        self.mouseMoved.emit(pos)

        self.prevMovePoint = pos
        self.restoreCursor()

        is_shift_pressed = ev.modifiers() & Qt.KeyboardModifier.ShiftModifier

        if (Qt.MouseButton.MiddleButton & ev.buttons()) and self._dragging:
            self.overrideCursor(CURSOR_MOVE)
            delta = self._drag_start_position - ev.globalPos()  # 计算位移
            self.moveRequest.emit(delta)
            self._drag_start_position = ev.globalPos()  # 更新当前位置
            self.repaint()
            return

        # Polygon drawing.
        if self.drawing():
            self.line.shape_type = self._determine_shape_type()

            self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                self.repaint()  # draw crosshair
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.createMode in ["polygon"]
                and self.closeEnough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)

            if self.createMode in ["polygon", "linestrip"]:
                self.line.points = [self.current[-1], pos]
                self.line.point_labels = [1, 1]
            elif self.createMode in ["ai_polygon", "ai_mask"]:
                self.line.points = [self.current.points[-1], pos]
                self.line.point_labels = [
                    self.current.point_labels[-1],
                    0 if is_shift_pressed else 1,
                ]
            elif self.createMode in ["rectangle", "barcode", "ai_barcode"]:
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.shape_type = "circle"
            elif self.createMode == "line":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "point":
                self.line.points = [self.current[0]]
                self.line.point_labels = [1]
                self.line.close()
            assert len(self.line.points) == len(self.line.point_labels)
            self.repaint()
            self.current.highlightClear()
            return

        # Polygon copy moving.
        if Qt.MouseButton.RightButton & ev.buttons():
            if self.selectedShapesCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if Qt.MouseButton.LeftButton & ev.buttons():
            if self._selecting:
                self.select_end = pos
                self.selectShapeRect(QtCore.QRectF(self.select_begin, self.select_end))
                self.repaint()
            else:
                if self.selectedVertex():
                    self.boundedMoveVertex(pos)
                    self.repaint()
                    self.movingShape = True
                elif self.selectedShapes and self.prevPoint:
                    self.overrideCursor(CURSOR_MOVE)
                    self.boundedMoveShapes(self.selectedShapes, pos)
                    self.repaint()
                    self.movingShape = True
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon)
            index_edge = shape.nearestEdge(pos, self.epsilon)
            if index is not None:
                if self.selectedVertex():
                    self.highlightShape.highlightClear()
                self.prevHighlightVertex = self.highlightVertex = index
                self.prevHighlightShape = self.highlightShape = shape
                self.prevHighlightEdge = self.highlightEdge
                self.highlightEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(
                    self.tr(
                        "Click & Drag to move point\n"
                        "ALT + SHIFT + Click to delete point"
                    )
                )
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif index_edge is not None and shape.canAddPoint():
                if self.selectedVertex():
                    self.highlightShape.highlightClear()
                self.prevHighlightVertex = self.highlightVertex
                self.highlightVertex = None
                self.prevHighlightShape = self.highlightShape = shape
                self.prevHighlightEdge = self.highlightEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("ALT + Click to create point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.highlightShape.highlightClear()
                self.prevHighlightVertex = self.highlightVertex
                self.highlightVertex = None
                self.prevHighlightShape = self.highlightShape = shape
                self.prevHighlightEdge = self.highlightEdge
                self.highlightEdge = None
                self.setToolTip(
                    self.tr("Click & drag to move shape '%s'") % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
        self.vertexSelected.emit(self.highlightVertex is not None)

    def addPointToEdge(self):
        shape = self.prevHighlightShape
        index = self.prevHighlightEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.highlightShape = shape
        self.highlightVertex = index
        self.highlightEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevHighlightShape
        index = self.prevHighlightVertex
        if shape is None or index is None:
            return
        shape.removePoint(index)
        shape.highlightClear()
        self.highlightShape = shape
        self.prevHighlightVertex = None
        self.movingShape = True  # Save changes

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        pos = self.transformPos(ev.localPos())

        is_shift_pressed = ev.modifiers() & Qt.KeyboardModifier.ShiftModifier

        if ev.button() == Qt.MouseButton.LeftButton:
            if self.drawing():
                if self.current:
                    # Add point to existing shape.
                    if self.createMode in ["polygon"]:
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in [
                        "barcode",
                        "ai_barcode",
                        "rectangle",
                        "circle",
                        "line",
                    ]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        if self.createMode in ["barcode"]:
                            shapes = _update_shape_with_decoder(
                                shape=self.current,
                                createMode=self.createMode,
                                image=self.pixmap.toImage(),
                            )
                            if len(shapes) == 0:
                                self.finalise()
                                return
                            for shape in shapes:
                                self.current = shape
                                self.finalise()
                        elif self.createMode in ["ai_barcode"]:
                            shapes = _update_shape_with_yolo_decoder(
                                shape=self.current,
                                createMode=self.createMode,
                                image=self.pixmap.toImage(),
                            )
                            if len(shapes) == 0:
                                self.finalise()
                                return
                            for shape in shapes:
                                self.current = shape
                                self.finalise()
                        else:
                            self.finalise()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == Qt.KeyboardModifier.ControlModifier:
                            self.finalise()
                    elif self.createMode in ["ai_polygon", "ai_mask"]:
                        self.current.addPoint(
                            self.line.points[1],
                            label=self.line.point_labels[1],
                        )
                        self.line.points[0] = self.current.points[-1]
                        self.line.point_labels[0] = self.current.point_labels[-1]
                        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    self.current = Shape(shape_type=self._determine_shape_type())
                    self.current.addPoint(pos, label=0 if is_shift_pressed else 1)
                    if self.createMode == "point":
                        self.finalise()  # 如果是点直接结束
                    elif (
                        self.createMode in ["ai_polygon", "ai_mask"]
                        and ev.modifiers() & Qt.KeyboardModifier.ControlModifier
                    ):
                        self.finalise()  # Control + LeftClick 结束
                    else:
                        if self.createMode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [pos, pos]
                        if (
                            self.createMode in ["ai_polygon", "ai_mask"]
                            and is_shift_pressed
                        ):
                            self.line.point_labels = [0, 0]
                        else:
                            self.line.point_labels = [1, 1]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            elif self.editing():
                if (
                    self.selectedEdge()
                    and ev.modifiers() == Qt.KeyboardModifier.AltModifier
                ):
                    self.addPointToEdge()
                elif self.selectedVertex() and ev.modifiers() == (
                    Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.ShiftModifier
                ):
                    self.removeSelectedPoint()

                group_mode = int(ev.modifiers()) == Qt.KeyboardModifier.ControlModifier
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos

                if (
                    len(self.selectedShapes) == 0
                    and self.prevPoint
                    and not self.selectedVertex()
                ):
                    self.select_begin = pos
                    self.select_end = self.select_begin
                    self._selecting = True

                self.repaint()

        elif ev.button() == Qt.MouseButton.RightButton and self.editing():
            group_mode = int(ev.modifiers()) == Qt.KeyboardModifier.ControlModifier
            if not self.selectedShapes or (
                self.highlightShape is not None
                and self.highlightShape not in self.selectedShapes
            ):
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.repaint()
            self.prevPoint = pos
        elif ev.button() == Qt.MouseButton.MiddleButton:
            self.overrideCursor(CURSOR_MOVE)
            self._drag_start_position = ev.globalPos()  # 记录开始拖动的位置
            self._dragging = True

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == Qt.MouseButton.RightButton:
            menu = self.menus[len(self.selectedShapesCopy) > 0]
            self.restoreCursor()
            if not menu.exec_(self.mapToGlobal(ev.pos())) and self.selectedShapesCopy:
                # Cancel the move by deleting the shadow copy.
                self.selectedShapesCopy = []
                self.repaint()
        elif ev.button() == Qt.MouseButton.LeftButton:
            self.select_begin = self.select_end = QtCore.QPoint()
            self._selecting = False
            self.update()

            if self.editing():
                if (
                    self.highlightShape is not None
                    and self.highlightShapeIsSelected
                    and not self.movingShape
                ):
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.highlightShape]
                    )
        elif ev.button() == Qt.MouseButton.MiddleButton:
            self._dragging = False
            self.restoreCursor()

        if self.movingShape and self.highlightShape:
            index = self.shapes.index(self.highlightShape)
            if self.shapesBackups[-1][index].points != self.shapes[index].points:
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and (
            (self.current and len(self.current) > 2)
            or self.createMode in ["ai_polygon", "ai_mask"]
        )

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent):
        if self.double_click != "close":
            return

        if (
            self.createMode in ["polygon"] and self.canCloseShape()
        ) or self.createMode in ["ai_polygon", "ai_mask"]:
            self.finalise()

    def selectShapes(self, shapes: list[Shape]):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode: bool):
        """Select the first shape created which contains this point."""
        # if self.selectedVertex():  # A vertex is marked for selection.
        #     index, shape = self.hVertex, self.hShape
        #     shape.highlightVertex(index, shape.MOVE_VERTEX)
        # else:
        for shape in reversed(self.shapes):
            if self.isVisible(shape) and shape.containsPoint(point):
                self.setHiding()
                if shape not in self.selectedShapes:
                    if multiple_selection_mode:
                        self.selectionChanged.emit(self.selectedShapes + [shape])
                    else:
                        self.selectionChanged.emit([shape])
                    self.highlightShapeIsSelected = False
                else:
                    self.highlightShapeIsSelected = True
                self.calculateOffsets(point)
                return
        self.deSelectShape()

    def selectShapeRect(self, rect: QtCore.QRectF):
        selectedShapes = self.selectedShapes

        def containsPoints(rect: QtCore.QRectF, points: QtCore.QPointF):
            for point in points:
                if not rect.contains(point):
                    return False
            return True

        for shape in reversed(self.shapes):
            if self.isVisible(shape) and (
                rect.contains(shape.boundingRect())
                or containsPoints(rect, shape.points)
            ):
                self.setHiding()
                if shape not in self.selectedShapes:
                    selectedShapes.append(shape)
                    self.highlightShapeIsSelected = False
                else:
                    self.highlightShapeIsSelected = True
        self.selectionChanged.emit(selectedShapes)
        self.calculateOffsets(rect.center())

    def calculateOffsets(self, point):
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.highlightVertex, self.highlightShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes: list[Shape], pos: QtCore.QPointF) -> bool:
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPointF(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPointF(
                min(0, self.pixmap.width() - o2.x()),
                min(0, self.pixmap.height() - o2.y()),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.highlightShapeIsSelected = False
            self.update()

    def deleteSelected(self) -> list[Shape]:
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def deleteShape(self, shape: list[Shape]):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.storeShapes()
        self.update()

    def paintEvent(self, event: Optional[QtGui.QPaintEvent]) -> None:
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)

        p.scale(1 / self.scale, 1 / self.scale)

        # 多选框
        if self.editing() and self.select_begin != self.select_end:
            b = p.brush()
            p.setPen(QtGui.QColor(0, 95, 184))
            p.setBrush(QtGui.QBrush(QtGui.QColor(0, 95, 184, 100)))
            p.drawRect(
                int(self.select_begin.x() * self.scale),
                int(self.select_begin.y() * self.scale),
                int((self.select_end.x() - self.select_begin.x()) * self.scale),
                int((self.select_end.y() - self.select_begin.y()) * self.scale),
            )
            p.setBrush(b)

        # draw crosshair
        if (
            self._crosshair[self._createMode]
            and self.drawing()
            and self.prevMovePoint
            and not self.outOfPixmap(self.prevMovePoint)
        ):
            p.setPen(QtGui.QColor(0, 0, 0))
            p.drawLine(
                0,
                int(self.prevMovePoint.y() * self.scale),
                self.width() - 1,
                int(self.prevMovePoint.y() * self.scale),
            )
            p.drawLine(
                int(self.prevMovePoint.x() * self.scale),
                0,
                int(self.prevMovePoint.x() * self.scale),
                self.height() - 1,
            )

        Shape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and self.isVisible(shape):
                shape.fill = shape.selected or shape == self.highlightShape
                shape.paint(p)
        if self.current:
            self.current.paint(p)
            assert len(self.line.points) == len(self.line.point_labels)
            self.line.paint(p)
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        if not self.current:
            p.end()
            return

        if (
            self.createMode in ["polygon"]
            and self.fillDrawing()
            and len(self.current.points) >= 2
        ):
            drawing_shape = self.current.copy()
            if drawing_shape.fill_color.getRgb()[3] == 0:
                logger.warning(
                    "fill_drawing=true, but fill_color is transparent,"
                    " so forcing to be opaque."
                )
                drawing_shape.fill_color.setAlpha(64)
            drawing_shape.addPoint(self.line[1])

        if self.createMode not in ["ai_polygon", "ai_mask"]:
            p.end()
            return

        drawing_shape = self.current.copy()
        drawing_shape.addPoint(
            point=self.line.points[1],
            label=self.line.point_labels[1],
        )
        if self.createMode in ["ai_polygon", "ai_mask"]:
            if self._sam is None:
                logger.warning("SAM model is not set yet")
                p.end()
                return
            _update_shape_with_sam(
                shape=drawing_shape,
                createMode=self.createMode,
                model_name=self._sam.name,
                image_embedding=self._sam_embedding[
                    labelme.utils.img_qt_to_arr(self.pixmap.toImage()).tobytes()
                ],
            )
        drawing_shape.fill = self.fillDrawing()
        drawing_shape.selected = True
        drawing_shape.paint(p)
        p.end()

    def transformPos(self, point: QtCore.QPointF) -> QtCore.QPointF:
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self) -> QtCore.QPointF:
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)

    def outOfPixmap(self, p: QtCore.QPoint) -> bool:
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        """Finalize the current shape."""
        assert self.current
        if self._sam:
            _update_shape_with_sam(
                shape=self.current,
                createMode=self.createMode,
                model_name=self._sam.name,
                image_embedding=self._sam_embedding[
                    labelme.utils.img_qt_to_arr(self.pixmap.toImage()).tobytes()
                ],
            )
        self.current.close()

        self.shapes.append(self.current)
        self.storeShapes()
        self.newShape.emit(self.current)
        self.current = None
        self.setHiding(False)
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPointF(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev: QtGui.QWheelEvent):
        mods = ev.modifiers()
        delta = ev.angleDelta()
        if Qt.KeyboardModifier.ControlModifier == int(mods):
            # with Ctrl/Command key
            # zoom
            self.zoomRequest.emit(delta.y(), ev.pos())
        else:
            # scroll
            self.scrollRequest.emit(delta.x(), Qt.Orientation.Horizontal)
            self.scrollRequest.emit(delta.y(), Qt.Orientation.Vertical)
        ev.accept()

    def moveByKeyboard(self, offset):
        if self.selectedShapes:
            self.boundedMoveShapes(self.selectedShapes, self.prevPoint + offset)
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        modifiers = ev.modifiers()
        key = ev.key()
        if self.drawing():
            if key == Qt.Key.Key_Escape and self.current:
                self.current = None
                self.drawingPolygon.emit(False)
                self.update()
            elif key == Qt.Key.Key_Return and self.canCloseShape():
                self.finalise()
            elif modifiers == Qt.KeyboardModifier.AltModifier:
                self.snapping = False
        elif self.editing():
            if key == Qt.Key.Key_Up:
                self.moveByKeyboard(QtCore.QPointF(0.0, -MOVE_SPEED))
            elif key == Qt.Key.Key_Down:
                self.moveByKeyboard(QtCore.QPointF(0.0, MOVE_SPEED))
            elif key == Qt.Key.Key_Left:
                self.moveByKeyboard(QtCore.QPointF(-MOVE_SPEED, 0.0))
            elif key == Qt.Key.Key_Right:
                self.moveByKeyboard(QtCore.QPointF(MOVE_SPEED, 0.0))

    def keyReleaseEvent(self, ev: QtGui.QKeyEvent):
        modifiers = ev.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            if self.movingShape and self.selectedShapes:
                index = self.shapes.index(self.selectedShapes[0])
                if self.shapesBackups[-1][index].points != self.shapes[index].points:
                    self.storeShapes()
                    self.shapeMoved.emit()

                self.movingShape = False

    def setLastLabel(self, text, flags):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        self.current.restoreShapeRaw()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode == "point":
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.pixmap = pixmap
        if self._sam:
            self._compute_and_cache_image_embedding()
        if clear_shapes:
            self.shapes = []
        self.update()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.highlightShape = None
        self.highlightVertex = None
        self.highlightEdge = None
        self.update()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None  # type: ignore[assignment]
        self.shapesBackups = []
        self.update()


def _update_shape_with_decoder(
    shape: Shape,
    createMode: str,
    image: QtGui.QImage,
) -> list[Shape]:
    if createMode not in ["barcode"]:
        raise ValueError(f"createMode must be 'barcode', not {createMode}")

    shapes: list[Shape] = []
    results = decoder.decode_barcode(image=image.copy(shape.boundingRect().toRect()))

    if results is None and len(results) == 0:
        logger.warning("No points returned by decoder")
        return shapes

    p = shape.boundingRect().toRect().topLeft()
    results = [[point + p for point in sublist] for sublist in results]

    # 获取当前时间
    now = datetime.now()
    # 将当前时间转换为时间戳
    timestamp = now.timestamp() * 1000
    for points in results:
        datamatrix = Shape(
            label="datamatrix",
            shape_type="polygon",
            group_id=int(timestamp),
        )
        datamatrix.addPoint(points[0])
        datamatrix.addPoint(points[1])
        datamatrix.addPoint(points[2])
        datamatrix.addPoint(points[3])
        shapes.append(datamatrix)
        bl = Shape(
            label="bl",
            shape_type="point",
            group_id=int(timestamp),
        )
        bl.addPoint(points[0])
        shapes.append(bl)
        br = Shape(
            label="br",
            shape_type="point",
            group_id=int(timestamp),
        )
        br.addPoint(points[1])
        shapes.append(br)
        tr = Shape(
            label="tr",
            shape_type="point",
            group_id=int(timestamp),
        )
        tr.addPoint(points[2])
        shapes.append(tr)
        tl = Shape(
            label="tl",
            shape_type="point",
            group_id=int(timestamp),
        )
        tl.addPoint(points[3])
        shapes.append(tl)
        timestamp += 1

    return shapes


def _update_shape_with_yolo_decoder(
    shape: Shape,
    createMode: str,
    image: QtGui.QImage,
) -> list[Shape]:
    if createMode not in ["ai_barcode"]:
        raise ValueError(f"createMode must be 'ai_barcode', not {createMode}")

    shapes: list[Shape] = []
    results = decoder.yolo_barcode(image.copy(shape.boundingRect().toRect()))

    if results is None and len(results) == 0:
        logger.warning("No points returned by decoder")
        return shapes

    p = shape.boundingRect().toRect().topLeft()
    results = [[point + p for point in sublist] for sublist in results]

    # 获取当前时间
    now = datetime.now()
    # 将当前时间转换为时间戳
    timestamp = now.timestamp() * 1000
    for points in results:
        datamatrix = Shape(
            label="datamatrix",
            shape_type="polygon",
            group_id=int(timestamp),
        )
        for point in points:
            datamatrix.addPoint(point)
        shapes.append(datamatrix)
        # bl = Shape(
        #     label="bl",
        #     shape_type="point",
        #     group_id=int(timestamp),
        # )
        # bl.addPoint(points[0])
        # shapes.append(bl)
        # br = Shape(
        #     label="br",
        #     shape_type="point",
        #     group_id=int(timestamp),
        # )
        # br.addPoint(points[1])
        # shapes.append(br)
        # tr = Shape(
        #     label="tr",
        #     shape_type="point",
        #     group_id=int(timestamp),
        # )
        # tr.addPoint(points[2])
        # shapes.append(tr)
        # tl = Shape(
        #     label="tl",
        #     shape_type="point",
        #     group_id=int(timestamp),
        # )
        # tl.addPoint(points[3])
        # shapes.append(tl)
        timestamp += 1

    return shapes


def _update_shape_with_sam(
    shape: Shape,
    createMode: str,
    model_name: str,
    image_embedding: osam.types.ImageEmbedding,
) -> None:
    if createMode not in ["ai_polygon", "ai_mask"]:
        raise ValueError(
            f"createMode must be 'ai_polygon' or 'ai_mask', not {createMode}"
        )

    response: osam.types.GenerateResponse = osam.apis.generate(
        osam.types.GenerateRequest(
            model=model_name,
            image_embedding=image_embedding,
            prompt=osam.types.Prompt(
                points=[[point.x(), point.y()] for point in shape.points],
                point_labels=shape.point_labels,
            ),
        )
    )
    if not response.annotations:
        logger.warning("No annotations returned by model {!r}", model_name)
        return

    if createMode == "ai_mask":
        y1: int
        x1: int
        y2: int
        x2: int
        if response.annotations[0].bounding_box is None:
            y1, x1, y2, x2 = imgviz.instances.mask_to_bbox(
                [response.annotations[0].mask]
            )[0].astype(int)
        else:
            y1 = response.annotations[0].bounding_box.ymin
            x1 = response.annotations[0].bounding_box.xmin
            y2 = response.annotations[0].bounding_box.ymax
            x2 = response.annotations[0].bounding_box.xmax
        shape.setShapeRefined(
            shape_type="mask",
            points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
            point_labels=[1, 1],
            mask=response.annotations[0].mask[y1 : y2 + 1, x1 : x2 + 1],
        )
    elif createMode == "ai_polygon":
        points = polygon_from_mask.compute_polygon_from_mask(
            mask=response.annotations[0].mask
        )
        if len(points) < 2:
            return
        shape.setShapeRefined(
            shape_type="polygon",
            points=[QtCore.QPointF(point[0], point[1]) for point in points],
            point_labels=[1] * len(points),
        )
