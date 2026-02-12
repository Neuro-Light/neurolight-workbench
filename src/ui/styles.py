"""NeuroLight application stylesheet - clean, professional UI for neuroscience workflows."""

# Typography - cross-platform (no Segoe UI; uses system fonts on macOS, Windows, Linux)
FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif"
FONT_SIZE_BASE = "13px"
FONT_SIZE_SMALL = "12px"
FONT_SIZE_LARGE = "14px"
FONT_SIZE_TITLE = "16px"
FONT_SIZE_DIALOG_TITLE = "22px"
FONT_SIZE_SECTION_HEADING = "15px"

# Spacing & radius
RADIUS = "6px"
RADIUS_SMALL = "4px"
RADIUS_LARGE = "8px"
PADDING = "8px 12px"
PADDING_SMALL = "4px 8px"

# Color palettes
COLORS_LIGHT = {
    # Light palette
    "bg": "#F5F5F5",
    # Surfaces / cards / inputs
    "surface": "#FFFFFF",
    # Alternate surface / subtle panels
    "surface_alt": "#FFFFFF",
    # Primary accent
    "primary": "#4A90E2",
    "primary_hover": "#357ABD",
    "primary_pressed": "#2D6FB3",
    # Text
    "text": "#212121",
    "text_secondary": "#616161",
    "text_disabled": "#9E9E9E",
    # Borders / separators
    "border": "#E0E0E0",
    "border_focus": "#4A90E2",
    # Hover / disabled fills
    "hover": "#EEEEEE",
    "disabled": "#E0E0E0",
    "success": "#059669",
    "warning": "#d97706",
    # Matplotlib toolbar: dark bg in light mode so white icons are visible
    "mpl_toolbar_bg": "#334155",
    "mpl_toolbar_text": "#f1f5f9",
    "mpl_toolbar_border": "#475569",
    "mpl_toolbar_hover": "#475569",
}

COLORS_DARK = {
    # Dark palette
    "bg": "#1E1E1E",
    # Surfaces / cards / inputs
    "surface": "#2D2D2D",  # Graph background / main surfaces
    "surface_alt": "#262626",  # Slightly darker alt surface
    # Primary accent
    "primary": "#4A90E2",
    "primary_hover": "#6AA7E8",
    "primary_pressed": "#357ABD",
    # Text
    "text": "#E0E0E0",
    "text_secondary": "#B0B0B0",
    "text_disabled": "#7A7A7A",
    # Borders / separators
    "border": "#3A3A3A",
    "border_focus": "#4A90E2",
    # Hover / disabled fills
    "hover": "#333333",
    "disabled": "#3A3A3A",
    "success": "#10b981",
    "warning": "#f59e0b",
}

# High contrast palettes (stronger text/background and border contrast)
COLORS_LIGHT_HC = {
    "bg": "#FFFFFF",
    "surface": "#FFFFFF",
    "surface_alt": "#F5F5F5",
    "primary": "#0066CC",
    "primary_hover": "#0052A3",
    "primary_pressed": "#003D7A",
    "text": "#000000",
    "text_secondary": "#000000",
    "text_disabled": "#595959",
    "border": "#000000",
    "border_focus": "#0066CC",
    "hover": "#E8E8E8",
    "disabled": "#CCCCCC",
    "success": "#0D7D4D",
    "warning": "#B35C00",
    "mpl_toolbar_bg": "#1A1A1A",
    "mpl_toolbar_text": "#FFFFFF",
    "mpl_toolbar_border": "#333333",
    "mpl_toolbar_hover": "#333333",
}

COLORS_DARK_HC = {
    "bg": "#000000",
    "surface": "#1A1A1A",
    "surface_alt": "#262626",
    "primary": "#6BB3FF",
    "primary_hover": "#8FC5FF",
    "primary_pressed": "#4A90E2",
    "text": "#FFFFFF",
    "text_secondary": "#FFFFFF",
    "text_disabled": "#999999",
    "border": "#FFFFFF",
    "border_focus": "#6BB3FF",
    "hover": "#262626",
    "disabled": "#333333",
    "success": "#00E676",
    "warning": "#FFB74D",
}


# Valid theme values (single selection in Preferences)
THEME_DARK = "dark"
THEME_LIGHT = "light"
THEME_DARK_HIGH_CONTRAST = "dark_high_contrast"
THEME_LIGHT_HIGH_CONTRAST = "light_high_contrast"


def _palette(theme: str) -> dict:
    """Return the color palette for the given theme.
    theme is one of: dark, light, dark_high_contrast, light_high_contrast.
    """
    if theme == THEME_DARK_HIGH_CONTRAST:
        return COLORS_DARK_HC
    if theme == THEME_LIGHT_HIGH_CONTRAST:
        return COLORS_LIGHT_HC
    if theme == THEME_LIGHT:
        return COLORS_LIGHT
    return COLORS_DARK


def get_mpl_theme(theme: str = "dark") -> dict:
    """Return matplotlib-friendly theme colors (facecolor, text, grid, line colors).
    Use for Figure/axes to match app theme (dark, light, dark_high_contrast, light_high_contrast).
    """
    c = _palette(theme)
    is_dark = theme in (THEME_DARK, THEME_DARK_HIGH_CONTRAST)
    is_high_contrast = theme in (THEME_DARK_HIGH_CONTRAST, THEME_LIGHT_HIGH_CONTRAST)
    if is_dark:
        return {
            "figure_facecolor": c["surface"],
            "axes_facecolor": c["surface"],
            "axes_edgecolor": c["border"],
            "text_color": c["text"],
            "grid_color": c["border"],
            "legend_facecolor": c["surface_alt"],
            "legend_edgecolor": c["border"],
            "good_color": "#22c55e",
            "bad_color": "#f87171",
            "neutral_color": "#6BB3FF" if is_high_contrast else "#60a5fa",
            "average_color": "#e2e8f0",
            "average_good_color": "#4ade80",
            "roi_line_color": "#6BB3FF" if is_high_contrast else "#60a5fa",
        }
    else:
        return {
            "figure_facecolor": c["surface"],
            "axes_facecolor": c["surface"],
            "axes_edgecolor": c["border"],
            "text_color": c["text"],
            "grid_color": c["border"],
            "legend_facecolor": c["surface_alt"],
            "legend_edgecolor": c["border"],
            "good_color": "green",
            "bad_color": "red",
            "neutral_color": "blue",
            "average_color": "black",
            "average_good_color": "darkgreen",
            "roi_line_color": "blue",
        }


def get_stylesheet(theme: str = "dark") -> str:
    """Return the application stylesheet string.

    Args:
        theme: One of "dark", "light", "dark_high_contrast", "light_high_contrast".
    """
    c = _palette(theme)
    return f"""
    /* === Global / Base === */
    QWidget {{
        background-color: {c["bg"]};
        color: {c["text"]};
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZE_BASE};
    }}

    QMainWindow {{
        background-color: {c["bg"]};
    }}

    /* === Dialogs === */
    QDialog {{
        background-color: {c["surface"]};
        color: {c["text"]};
    }}

    /* Experiment Manager: same darker background as main app so buttons/experiments pop */
    QDialog#experimentManagerDialog {{
        background-color: {c["bg"]};
    }}

    /* === Labels === */
    QLabel {{
        color: {c["text"]};
        font-size: {FONT_SIZE_BASE};
        background-color: transparent;
    }}

    QLabel[class="title"] {{
        font-size: {FONT_SIZE_TITLE};
        font-weight: 600;
        color: {c["text"]};
        background-color: transparent;
    }}

    /* Dialog / app title (e.g. Experiment Manager) - bold, large, no highlight bar */
    QLabel[class="dialog-title"] {{
        font-size: {FONT_SIZE_DIALOG_TITLE};
        font-weight: 700;
        color: {c["text"]};
        background-color: transparent;
    }}

    /* Section heading (e.g. Recent Experiments) - bold, centered */
    QLabel[class="section-heading"] {{
        font-size: {FONT_SIZE_SECTION_HEADING};
        font-weight: 700;
        color: {c["text"]};
        background-color: transparent;
    }}

    /* === Buttons === */
    QPushButton {{
        background-color: {c["surface"]};
        color: {c["text"]};
        border: 1px solid {c["border"]};
        border-radius: {RADIUS};
        padding: {PADDING};
        font-size: {FONT_SIZE_BASE};
        min-height: 24px;
    }}

    QPushButton:hover {{
        background-color: {c["hover"]};
        border-color: {c["primary"]};
        color: {c["primary"]};
    }}

    QPushButton:pressed {{
        background-color: {c["border"]};
        border-color: {c["primary_pressed"]};
    }}

    QPushButton:disabled {{
        background-color: {c["surface_alt"]};
        color: {c["text_disabled"]};
        border-color: {c["border"]};
    }}

    /* Primary action buttons (e.g., Open Images, Detect Neurons) */
    QPushButton[class="primary"] {{
        background-color: {c["primary"]};
        color: white;
        border: 1px solid {c["primary"]};
    }}

    QPushButton[class="primary"]:hover {{
        background-color: {c["primary_hover"]};
        border-color: {c["primary_hover"]};
        color: white;
    }}

    QPushButton[class="primary"]:pressed {{
        background-color: {c["primary_pressed"]};
        border-color: {c["primary_pressed"]};
    }}

    QPushButton[class="primary"]:disabled {{
        background-color: {c["disabled"]};
        border-color: {c["disabled"]};
        color: {c["text_disabled"]};
    }}

    /* Buttons that match selected tab (surface bg, primary text) - e.g. Experiment Manager */
    QPushButton[class="tab-action"] {{
        background-color: {c["surface"]};
        color: {c["primary"]};
        border: 1px solid {c["border"]};
        font-weight: 600;
    }}

    QPushButton[class="tab-action"]:hover {{
        background-color: {c["hover"]};
        border-color: {c["primary"]};
        color: {c["primary"]};
    }}

    QPushButton[class="tab-action"]:pressed {{
        background-color: {c["border"]};
        border-color: {c["primary"]};
        color: {c["primary"]};
    }}

    QPushButton[class="tab-action"]:disabled {{
        background-color: {c["surface_alt"]};
        color: {c["text_disabled"]};
        border-color: {c["border"]};
    }}

    /* Same look for QToolButton used as row options (e.g. ...) */
    QToolButton[class="tab-action"] {{
        background-color: {c["surface"]};
        color: {c["primary"]};
        border: 1px solid {c["border"]};
        font-weight: 600;
    }}

    QToolButton[class="tab-action"]:hover {{
        background-color: {c["hover"]};
        border-color: {c["primary"]};
        color: {c["primary"]};
    }}

    QToolButton[class="tab-action"]:pressed {{
        background-color: {c["border"]};
        border-color: {c["primary"]};
        color: {c["primary"]};
    }}

    QToolButton[class="tab-action"]:disabled {{
        background-color: {c["surface_alt"]};
        color: {c["text_disabled"]};
        border-color: {c["border"]};
    }}

    /* === Line Edits & Text === */
    QLineEdit, QPlainTextEdit {{
        background-color: {c["surface"]};
        color: {c["text"]};
        border: 1px solid {c["border"]};
        border-radius: {RADIUS_SMALL};
        padding: 6px 10px;
        font-size: {FONT_SIZE_BASE};
        selection-background-color: {c["primary"]};
        selection-color: white;
    }}

    QLineEdit:focus, QPlainTextEdit:focus {{
        border-color: {c["border_focus"]};
    }}

    QLineEdit:disabled, QPlainTextEdit:disabled {{
        background-color: {c["surface_alt"]};
        color: {c["text_disabled"]};
    }}

    /* === Spin Boxes === */
    QSpinBox, QDoubleSpinBox {{
        background-color: {c["surface"]};
        color: {c["text"]};
        border: 1px solid {c["border"]};
        border-radius: {RADIUS_SMALL};
        padding: 6px 10px;
        font-size: {FONT_SIZE_BASE};
        min-width: 60px;
    }}

    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {c["border_focus"]};
    }}

    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        background-color: {c["surface_alt"]};
        border: none;
        width: 18px;
        border-radius: {RADIUS_SMALL};
    }}

    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {c["hover"]};
    }}

    /* === Combo Box === */
    QComboBox {{
        background-color: {c["surface"]};
        color: {c["text"]};
        border: 1px solid {c["border"]};
        border-radius: {RADIUS_SMALL};
        padding: 6px 10px;
        font-size: {FONT_SIZE_BASE};
        min-height: 22px;
    }}

    QComboBox:hover {{
        border-color: {c["primary"]};
    }}

    QComboBox:focus {{
        border-color: {c["border_focus"]};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 24px;
        background-color: {c["surface_alt"]};
        border-radius: 0 {RADIUS_SMALL} {RADIUS_SMALL} 0;
    }}

    QComboBox::down-arrow {{
        width: 10px;
        height: 10px;
        border: 2px solid {c["text_secondary"]};
        border-width: 0 2px 2px 0;
        margin: 4px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {c["surface"]};
        border: 1px solid {c["border"]};
        border-radius: {RADIUS_SMALL};
        padding: 4px;
        selection-background-color: {c["primary"]};
        selection-color: white;
    }}

    /* === Check Box === */
    QCheckBox {{
        color: {c["text"]};
        font-size: {FONT_SIZE_BASE};
        spacing: 8px;
    }}

    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {c["border"]};
        border-radius: {RADIUS_SMALL};
        background-color: {c["surface"]};
    }}

    QCheckBox::indicator:hover {{
        border-color: {c["primary"]};
    }}

    QCheckBox::indicator:checked {{
        background-color: {c["primary"]};
        border-color: {c["primary"]};
    }}

    QCheckBox::indicator:disabled {{
        background-color: {c["surface_alt"]};
        border-color: {c["disabled"]};
    }}

    /* === Radio Button (same blue box as checkbox for theme selection) === */
    QRadioButton {{
        color: {c["text"]};
        font-size: {FONT_SIZE_BASE};
        spacing: 8px;
    }}

    QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {c["border"]};
        border-radius: {RADIUS_SMALL};
        background-color: {c["surface"]};
    }}

    QRadioButton::indicator:hover {{
        border-color: {c["primary"]};
    }}

    QRadioButton::indicator:checked {{
        background-color: {c["primary"]};
        border-color: {c["primary"]};
    }}

    QRadioButton::indicator:disabled {{
        background-color: {c["surface_alt"]};
        border-color: {c["disabled"]};
    }}

    /* === Sliders === */
    QSlider::groove:horizontal {{
        height: 6px;
        background-color: {c["border"]};
        border-radius: 3px;
    }}

    QSlider::handle:horizontal {{
        width: 16px;
        height: 16px;
        margin: -5px 0;
        background-color: {c["surface"]};
        border: 2px solid {c["border"]};
        border-radius: 8px;
    }}

    QSlider::handle:horizontal:hover {{
        border-color: {c["primary"]};
        background-color: {c["hover"]};
    }}

    QSlider::sub-page:horizontal {{
        background-color: {c["primary"]};
        border-radius: 3px;
    }}

    /* === Group Box === */
    QGroupBox {{
        font-size: {FONT_SIZE_BASE};
        font-weight: 600;
        color: {c["text"]};
        border: 1px solid {c["border"]};
        border-radius: {RADIUS};
        margin-top: 12px;
        padding: 16px 12px 12px 12px;
        padding-top: 24px;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        top: 4px;
        padding: 0 6px;
        background-color: {c["surface"]};
        color: {c["text_secondary"]};
        font-weight: 600;
        font-size: {FONT_SIZE_SMALL};
    }}

    /* === Tab Widget === */
    QTabWidget::pane {{
        border: 1px solid {c["border"]};
        border-radius: {RADIUS};
        background-color: {c["surface"]};
        top: -1px;
        padding: 8px;
    }}

    QTabBar::tab {{
        background-color: {c["surface_alt"]};
        color: {c["text_secondary"]};
        border: 1px solid {c["border"]};
        border-bottom: none;
        border-top-left-radius: {RADIUS};
        border-top-right-radius: {RADIUS};
        padding: 10px 20px;
        margin-right: 2px;
        font-size: {FONT_SIZE_BASE};
    }}

    QTabBar::tab:selected {{
        background-color: {c["surface"]};
        color: {c["primary"]};
        font-weight: 600;
        border-color: {c["border"]};
        border-bottom: 1px solid {c["surface"]};
        margin-bottom: -1px;
    }}

    QTabBar::tab:hover:!selected {{
        background-color: {c["hover"]};
        color: {c["text"]};
    }}

    /* === List Widget === */
    QListWidget {{
        background-color: {c["surface"]};
        border: 1px solid {c["border"]};
        border-radius: {RADIUS};
        padding: 4px;
        outline: none;
    }}

    QListWidget::item {{
        padding: 10px 12px;
        border-radius: {RADIUS_SMALL};
        background-color: {c["surface"]};
        border: 1px solid {c["border"]};
        margin: 3px 4px;
        min-height: 20px;
    }}

    QListWidget::item:hover {{
        background-color: {c["hover"]};
        border-color: {c["primary"]};
    }}

    QListWidget::item:selected {{
        background-color: {c["primary"]};
        color: white;
        border-color: {c["primary"]};
    }}

    QListWidget::item:selected:!active {{
        background-color: {c["surface_alt"]};
        color: {c["text"]};
        border-color: {c["border"]};
    }}

    /* Horizontal divider between Load button and Recent Experiments (Experiment Manager) */
    QFrame#experimentManagerDivider {{
        background-color: {c["border"]};
        border: none;
    }}

    /* Recent experiment row widget - transparent so item border/card shows */
    QWidget#recentExperimentRow {{
        background-color: transparent;
    }}

    /* === Splitter === */
    QSplitter::handle {{
        background-color: {c["border"]};
        width: 2px;
        height: 2px;
    }}

    QSplitter::handle:hover {{
        background-color: {c["primary"]};
    }}

    /* === Progress Bar === */
    QProgressBar {{
        border: 1px solid {c["border"]};
        border-radius: {RADIUS};
        text-align: center;
        background-color: {c["surface_alt"]};
        height: 24px;
    }}

    QProgressBar::chunk {{
        background-color: {c["primary"]};
        border-radius: {RADIUS_SMALL};
    }}

    /* === Text Edit (read-only log, etc.) === */
    QTextEdit {{
        background-color: {c["surface_alt"]};
        color: {c["text"]};
        border: 1px solid {c["border"]};
        border-radius: {RADIUS};
        padding: 8px;
        font-family: 'Menlo', 'Monaco', 'Consolas', monospace;
        font-size: {FONT_SIZE_SMALL};
        selection-background-color: {c["primary"]};
        selection-color: white;
    }}

    /* === Dialog Button Box === */
    QDialogButtonBox QPushButton {{
        min-width: 80px;
    }}

    /* === Menu Bar & Menus === */
    QMenuBar {{
        background-color: {c["surface"]};
        color: {c["text"]};
        border-bottom: 1px solid {c["border"]};
        padding: 4px 0;
    }}

    QMenuBar::item {{
        padding: 6px 12px;
        border-radius: {RADIUS_SMALL};
    }}

    QMenuBar::item:selected {{
        background-color: {c["hover"]};
        color: {c["primary"]};
    }}

    QMenu {{
        background-color: {c["surface"]};
        border: 1px solid {c["border"]};
        border-radius: {RADIUS};
        padding: 4px;
    }}

    QMenu::item {{
        padding: 8px 24px 8px 12px;
        border-radius: {RADIUS_SMALL};
    }}

    QMenu::item:selected {{
        background-color: {c["hover"]};
        color: {c["primary"]};
    }}

    QMenu::separator {{
        height: 1px;
        background-color: {c["border"]};
        margin: 4px 8px;
    }}

    /* === Status Bar === */
    QStatusBar {{
        background-color: {c["surface"]};
        color: {c["text_secondary"]};
        border-top: 1px solid {c["border"]};
        padding: 4px 8px;
        font-size: {FONT_SIZE_SMALL};
    }}

    /* === Tool Bar (matplotlib navigation toolbar) === */
    QToolBar {{
        background-color: {c["surface"]};
        color: {c["text"]};
        border: 1px solid {c["border"]};
        border-radius: {RADIUS};
        padding: 4px;
        spacing: 4px;
    }}

    QToolBar QToolButton {{
        background-color: {c["surface"]};
        color: {c["text"]};
        border: 1px solid transparent;
        border-radius: {RADIUS_SMALL};
        padding: 4px 8px;
    }}

    QToolBar QToolButton:hover {{
        background-color: {c["hover"]};
        color: {c["primary"]};
        border-color: {c["border"]};
    }}

    QToolBar QToolButton:pressed {{
        background-color: {c["border"]};
    }}

    /* Matplotlib toolbar: in light mode use dark bg so white icons are visible */
    QToolBar#mpl_nav_toolbar {{
        background-color: {c.get("mpl_toolbar_bg", c["surface"])};
        color: {c.get("mpl_toolbar_text", c["text"])};
        border: 1px solid {c.get("mpl_toolbar_border", c["border"])};
    }}

    QToolBar#mpl_nav_toolbar QToolButton {{
        background-color: {c.get("mpl_toolbar_bg", c["surface"])};
        color: {c.get("mpl_toolbar_text", c["text"])};
    }}

    QToolBar#mpl_nav_toolbar QToolButton:hover {{
        background-color: {c.get("mpl_toolbar_hover", c["hover"])};
        color: {c.get("mpl_toolbar_text", c["text"])};
    }}

    QToolBar#mpl_nav_toolbar QToolButton:pressed {{
        background-color: {c.get("mpl_toolbar_border", c["border"])};
    }}

    /* === Scroll Bars === */
    QScrollBar:vertical {{
        background-color: {c["surface_alt"]};
        width: 12px;
        border-radius: 6px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background-color: {c["disabled"]};
        border-radius: 6px;
        min-height: 24px;
        margin: 2px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {c["text_secondary"]};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    QScrollBar:horizontal {{
        background-color: {c["surface_alt"]};
        height: 12px;
        border-radius: 6px;
        margin: 0;
    }}

    QScrollBar::handle:horizontal {{
        background-color: {c["disabled"]};
        border-radius: 6px;
        min-width: 24px;
        margin: 2px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background-color: {c["text_secondary"]};
    }}

    /* === Form Layout spacing === */
    QFormLayout QLabel {{
        color: {c["text_secondary"]};
    }}

    /* Plot hover / coordinate readout label */
    QLabel[class="plot-hover"] {{
        color: {c["text_secondary"]};
        font-size: {FONT_SIZE_SMALL};
    }}

    /* === Image container (neutral background for image display) === */
    QWidget#imageContainer {{
        background-color: {c["surface_alt"]};
    }}

    /* === Tooltip === */
    QToolTip {{
        background-color: {c["text"]};
        color: {c["surface"]};
        border: none;
        border-radius: {RADIUS_SMALL};
        padding: 6px 10px;
        font-size: {FONT_SIZE_SMALL};
    }}

    /* === Workflow stepper (top process bar) === */
    QFrame[objectName="workflowStepper"] {{
        background-color: {c["surface"]};
        border-bottom: 1px solid {c["border"]};
    }}

    QFrame[objectName="workflowStepper"] QToolButton {{
        text-align: center;
        padding: 6px 10px;
        border-radius: {RADIUS};
        border: 1px solid {c["border"]};
        background-color: {c["surface_alt"]};
        color: {c["text_secondary"]};
        font-size: {FONT_SIZE_SMALL};
    }}

    /* Active/current step – emphasize with blue outline and stronger text */
    QFrame[objectName="workflowStepper"] QToolButton[workflowStatus="active"] {{
        border: 2px solid {c["primary"]};
        background-color: {c["surface"]};
        color: {c["primary"]};
        font-weight: 600;
    }}

    /* Completed steps – soft green accent with checkmark text already set in code */
    QFrame[objectName="workflowStepper"] QToolButton[workflowStatus="completed"] {{
        border: 1px solid {c["success"] if "success" in c else c["primary"]};
        background-color: {c["surface"]};
        color: {c["text"]};
    }}

    /* Locked/upcoming steps – dimmed */
    QFrame[objectName="workflowStepper"] QToolButton[workflowStatus="locked"] {{
        background-color: {c["surface_alt"]};
        color: {c["text_disabled"]};
        border-style: dashed;
    }}

    QFrame[objectName="workflowStepper"] QLabel {{
        color: {c["text_secondary"]};
        font-size: {FONT_SIZE_SMALL};
    }}

    /* === Message Box (kept consistent with theme) === */
    QMessageBox {{
        background-color: {c["surface"]};
    }}
    """
