import sys
import numpy as np
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QGroupBox, QSizePolicy, QGraphicsDropShadowEffect, QPushButton, QFrame)
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor, QScreen
from PyQt6.QtCore import Qt, pyqtSlot, QPropertyAnimation, QEasingCurve
import pyqtgraph as pg
from pyqtgraph import PlotWidget
import qtawesome as qta


from UI.ui_worker import ProcessingWorker, PlotUpdateWorker
# from ui_worker import ProcessingWorker, PlotUpdateWorker

# Change these colors to change the app's look
COLOR_BACKGROUND = "#1F2227"
COLOR_CONTENT_BACKGROUND = "#1e1f24"
COLOR_TEXT_PRIMARY = "#f5f6fa"
COLOR_TEXT_SECONDARY = "#dcdde1"
COLOR_ACCENT_PRIMARY = "#00a8ff"
COLOR_ACCENT_SECONDARY = "#4cd137"
COLOR_ACCENT_WARN = "#fbc531"
COLOR_ACCENT_ALERT = "#c22e2e"
COLOR_SHADOW = "#181a1f"

# Plotting colors
# Change these to modify plot line colors
PLOT_PEN_SIGNAL = pg.mkPen("#378ac2", width=2)
PLOT_PEN_HR = pg.mkPen(COLOR_ACCENT_ALERT, width=2)
PLOT_PEN_BR = pg.mkPen(COLOR_ACCENT_PRIMARY, width=2)
PLOT_PEN_SDNN = pg.mkPen(COLOR_ACCENT_SECONDARY, width=2)
PLOT_PEN_RMSSD = pg.mkPen(COLOR_ACCENT_WARN, width=2)


class AppWindow(QMainWindow):
    def __init__(self, logic_function):
        super().__init__()
        # Change window title here
        self.setWindowTitle("Contactless Health Monitoring")
        # Change window size here
        self.setGeometry(100, 100, 1500, 800)
        self._center_window()

        pg.setConfigOption('background', COLOR_CONTENT_BACKGROUND)
        pg.setConfigOption('foreground', COLOR_TEXT_SECONDARY)

        self.worker = None
        self.plot_worker = None
        self.start_time = None
        self.logic_function = logic_function

        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {COLOR_BACKGROUND}; font-family: "Roboto", Arial, sans-serif; }}
            QLabel {{ color: {COLOR_TEXT_SECONDARY}; background-color: transparent; }}
            QGroupBox {{ background-color: {COLOR_CONTENT_BACKGROUND}; border: none; border-radius: 8px; }}
            PlotWidget {{ border-radius: 8px; }}
            QPushButton {{ background-color: {COLOR_ACCENT_PRIMARY}; color: {COLOR_TEXT_PRIMARY}; border-radius: 5px; padding: 6px 12px; font-weight: bold; border: 1px solid {COLOR_ACCENT_PRIMARY}; }}
            QPushButton:hover {{ background-color: #0097e6; }}
            QPushButton:pressed {{ background-color: #0082c8; }}
            QPushButton:disabled {{ background-color: #576574; color: #a4b0be; border: 1px solid #576574; }}
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        self._init_ui_components()
        self.reset_monitoring()

    def _center_window(self):
        # Center the window on the screen
        screen = QScreen.availableGeometry(QApplication.primaryScreen())
        self.move(screen.center() - self.frameGeometry().center())

    def _apply_shadow_effect(self, widget):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(5)
        shadow.setColor(QColor(COLOR_SHADOW))
        widget.setGraphicsEffect(shadow)

    def _init_ui_components(self):
        left_column_widget = QWidget()
        left_column_layout = QVBoxLayout(left_column_widget)
        left_column_layout.setSpacing(15)
        left_column_layout.setContentsMargins(0, 0, 0, 0)

        icon_hr = qta.icon('fa5s.heartbeat', color=COLOR_ACCENT_ALERT)
        icon_br = qta.icon('fa5s.wind', color=COLOR_ACCENT_PRIMARY)
        icon_stress = qta.icon('fa5s.bolt', color=COLOR_ACCENT_WARN)
        self.icon_status_ok = qta.icon('fa5s.check-circle', color=COLOR_ACCENT_SECONDARY)
        self.icon_status_warn = qta.icon('fa5s.exclamation-triangle', color=COLOR_ACCENT_WARN)
        self.icon_status_search = qta.icon('fa5s.sync-alt', color=COLOR_ACCENT_PRIMARY, animation=qta.Spin(self))
        self.icon_status_ready = qta.icon('fa5s.power-off', color=COLOR_TEXT_SECONDARY)

        self.hr_group = self._create_metrics_group("Heart Rate (HR)", "hr", icon_hr)
        self.br_group = self._create_metrics_group("Breathing Rate (BR)", "br", icon_br)
        self.stress_group = self._create_metrics_group("Stress Level", "stress", icon_stress)
        self.info_panel = self._create_info_panel()

        left_column_layout.addWidget(self.hr_group)
        left_column_layout.addWidget(self.br_group)
        left_column_layout.addWidget(self.stress_group)
        left_column_layout.addWidget(self.info_panel)

        self.status_panel = self._create_status_panel()
        left_column_layout.addWidget(self.status_panel)
        left_column_layout.addStretch()

        right_column_widget = QWidget()
        right_column_layout = QVBoxLayout(right_column_widget)
        right_column_layout.setSpacing(15)
        right_column_layout.setContentsMargins(0, 0, 0, 0)

        video_group_box = QGroupBox("Live Video Feed")
        video_group_box.setStyleSheet("QGroupBox { color: white; font-weight: bold; border-radius: 8px; padding: 8px; }")
        video_layout = QVBoxLayout(video_group_box)
        self.video_display_label = QLabel("Press Start to Begin Monitoring")
        self.video_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display_label.setMinimumSize(440, 320)
        self.video_display_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_layout.addWidget(self.video_display_label)
        self._apply_shadow_effect(video_group_box)

        self.rppg_plot_widget = self._create_plot("rPPG Signal", "Amplitude")
        self.rppg_signal_curve = self.rppg_plot_widget.getPlotItem().plot(pen=PLOT_PEN_SIGNAL)
        self.rppg_plot_widget.getPlotItem().getAxis('left').setWidth(50)

        self.hr_plot_widget = self._create_plot("HR over Time", "HR (bpm)", PLOT_PEN_HR, y_range=(40, 180))
        self.hr_curve = self.hr_plot_widget.getPlotItem().listDataItems()[0]

        self.hrv_plot_widget = self._create_plot("HRV over Time", "HRV (ms)", y_range=(0, 200))
        self.hrv_plot_widget.addLegend(offset=(-10, 0))
        self.sdnn_curve = self.hrv_plot_widget.getPlotItem().plot(pen=PLOT_PEN_SDNN, name="🟢 SDNN")
        self.rmssd_curve = self.hrv_plot_widget.getPlotItem().plot(pen=PLOT_PEN_RMSSD, name="🟡 RMSSD")

        self.br_plot_widget = self._create_plot("BR over Time", "BR (brpm)", PLOT_PEN_BR, y_range=(6, 30))
        self.br_curve = self.br_plot_widget.getPlotItem().listDataItems()[0]

        right_column_layout.addWidget(video_group_box, stretch=2)
        right_column_layout.addWidget(self.rppg_plot_widget, stretch=1)
        right_column_layout.addWidget(self.hr_plot_widget, stretch=1)
        right_column_layout.addWidget(self.hrv_plot_widget, stretch=1)
        right_column_layout.addWidget(self.br_plot_widget, stretch=1)

        self.main_layout.addWidget(left_column_widget, stretch=1)
        self.main_layout.addWidget(right_column_widget, stretch=2)

    def _create_info_row(self, label_text):
        row_layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setFont(QFont("Segoe UI", 10))
        label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY};")

        value_label = QLabel("N/A")
        value_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        value_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY};")
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        row_layout.addWidget(label)
        row_layout.addWidget(value_label)
        return row_layout, value_label

    def _create_separator(self):
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet(f"border: 1px solid {COLOR_SHADOW};")
        return separator

    def _create_info_subheader(self, text):
        label = QLabel(text)
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        label.setFont(font)
        label.setStyleSheet(f"""
            color: {COLOR_TEXT_SECONDARY};
            background-color: {COLOR_SHADOW};
            border-radius: 4px;
            padding: 2px 8px;
            margin-top: 8px;
            margin-bottom: 4px;
        """)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _create_info_panel(self):
        info_group_box = QGroupBox()
        main_layout = QVBoxLayout(info_group_box)
        main_layout.setContentsMargins(15, 10, 15, 15)
        main_layout.setSpacing(8)

        title_label = QLabel("Info Panel")
        title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {COLOR_ACCENT_PRIMARY};")
        main_layout.addWidget(title_label)
        main_layout.addSpacing(5)

        subtle_glow_color = COLOR_TEXT_SECONDARY
        subtle_glow_params = {'blur_radius': 15, 'duration': 600}

        # HRV METRICS
        main_layout.addWidget(self._create_info_subheader("HEART RATE VARIABILITY"))
        sdnn_layout, self.sdnn_info_value_label = self._create_info_row("💓 SDNN:")
        self._setup_glow_effect(self.sdnn_info_value_label, COLOR_ACCENT_SECONDARY, **subtle_glow_params)
        rmssd_layout, self.rmssd_info_value_label = self._create_info_row("📈 RMSSD:")
        self._setup_glow_effect(self.rmssd_info_value_label, COLOR_ACCENT_WARN, **subtle_glow_params)
        main_layout.addLayout(sdnn_layout)
        main_layout.addLayout(rmssd_layout)

        # HEAD POSE & QUALITY
        main_layout.addWidget(self._create_info_subheader("SIGNAL & POSE"))
        fps_layout, self.fps_info_value_label = self._create_info_row("🎯 FPS:")
        self._setup_glow_effect(self.fps_info_value_label, subtle_glow_color, **subtle_glow_params)
        quality_score_layout, self.quality_score_value_label = self._create_info_row("📊 Quality Score:")
        self._setup_glow_effect(self.quality_score_value_label, subtle_glow_color, **subtle_glow_params)
        main_layout.addLayout(fps_layout)
        main_layout.addLayout(quality_score_layout)

        yaw_layout, self.yaw_info_value_label = self._create_info_row("↔️ Yaw:")
        self._setup_glow_effect(self.yaw_info_value_label, subtle_glow_color, **subtle_glow_params)
        pitch_layout, self.pitch_info_value_label = self._create_info_row("↕️ Pitch:")
        self._setup_glow_effect(self.pitch_info_value_label, subtle_glow_color, **subtle_glow_params)
        roll_layout, self.roll_info_value_label = self._create_info_row("🔄 Roll:")
        self._setup_glow_effect(self.roll_info_value_label, subtle_glow_color, **subtle_glow_params)
        main_layout.addLayout(yaw_layout)
        main_layout.addLayout(pitch_layout)
        main_layout.addLayout(roll_layout)

        # COGNITIVE METRICS
        main_layout.addWidget(self._create_info_subheader("COGNITIVE ANALYSIS"))
        blink_layout, self.blink_info_value_label = self._create_info_row("👁️ Blink Rate:")
        self._setup_glow_effect(self.blink_info_value_label, subtle_glow_color, **subtle_glow_params)
        blink_count_layout, self.blink_count_info_value_label = self._create_info_row("🔢 Blink Count:")
        self._setup_glow_effect(self.blink_count_info_value_label, subtle_glow_color, **subtle_glow_params)
        attention_info_layout, self.attention_info_value_label = self._create_info_row("🧠 Attention:")
        self._setup_glow_effect(self.attention_info_value_label, subtle_glow_color, **subtle_glow_params)
        gaze_layout, self.gaze_info_value_label = self._create_info_row("👀 Gaze:")
        self._setup_glow_effect(self.gaze_info_value_label, subtle_glow_color, **subtle_glow_params)
        cognitive_status_layout, self.gaze_status_info_value_label = self._create_info_row("🧩 Gaze State:")
        self._setup_glow_effect(self.gaze_status_info_value_label, subtle_glow_color, **subtle_glow_params)
        main_layout.addLayout(blink_layout)
        main_layout.addLayout(blink_count_layout)
        main_layout.addLayout(attention_info_layout)
        main_layout.addLayout(gaze_layout)
        main_layout.addLayout(cognitive_status_layout)

        self._apply_shadow_effect(info_group_box)
        return info_group_box

    def _create_status_panel(self):
        status_group_box = QGroupBox()
        main_layout = QVBoxLayout(status_group_box)
        main_layout.setContentsMargins(15, 10, 15, 15)
        main_layout.setSpacing(15)

        title_label = QLabel("System Status")
        title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {COLOR_ACCENT_PRIMARY};")

        status_layout = QHBoxLayout()
        self.status_icon_label = QLabel()
        self.status_text_label = QLabel("Ready")
        self.status_text_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        status_layout.addWidget(self.status_icon_label)
        status_layout.addWidget(self.status_text_label, stretch=1)

        summary_font = QFont("Segoe UI", 10)
        summary_value_font = QFont("Segoe UI", 10, QFont.Weight.Bold)

        def create_summary_row(label_text):
            row_layout = QHBoxLayout()
            label = QLabel(label_text)
            label.setFont(summary_font)
            value_label = QLabel("N/A")
            value_label.setFont(summary_value_font)
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            row_layout.addWidget(label)
            row_layout.addWidget(value_label)
            return row_layout, value_label

        avg_hr_layout, self.avg_hr_label = create_summary_row("Average Heart Rate:")
        avg_br_layout, self.avg_br_label = create_summary_row("Average Breathing Rate:")

        main_layout.addWidget(title_label)
        main_layout.addLayout(status_layout)
        main_layout.addLayout(avg_hr_layout)
        main_layout.addLayout(avg_br_layout)

        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.reset_button = QPushButton("Reset")
        self.exit_button = QPushButton("Exit")

        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.reset_button.clicked.connect(self.reset_monitoring)
        self.exit_button.clicked.connect(self.close)

        self.exit_button.setStyleSheet(f"background-color: {COLOR_ACCENT_ALERT}; border-color: {COLOR_ACCENT_ALERT};")

        for btn in [self.start_button, self.stop_button, self.reset_button, self.exit_button]:
            btn.setMinimumWidth(70)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addWidget(self.exit_button)
        buttons_layout.addStretch()
        main_layout.addLayout(buttons_layout)

        self._apply_shadow_effect(status_group_box)
        return status_group_box

    def _setup_glow_effect(self, widget, glow_color_hex, blur_radius=20, duration=800):
        # Temporary glow animation to a widget
        effect = QGraphicsDropShadowEffect(widget)
        effect.setOffset(0, 0)
        effect.setBlurRadius(blur_radius)
        effect.setColor(QColor(0, 0, 0, 0))
        widget.setGraphicsEffect(effect)

        animation = QPropertyAnimation(effect, b"color")
        animation.setDuration(duration)
        
        start_color = QColor(glow_color_hex)
        start_color.setAlpha(150)
        
        animation.setStartValue(start_color)
        animation.setEndValue(QColor(0, 0, 0, 0))
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        widget.glow_animation = animation

    def _update_label_and_glow(self, label, new_text):
        # Updates label text and triggers glow animation if value changed
        new_text_str = str(new_text)
        if label.text() != new_text_str:
            label.setText(new_text_str)
            if hasattr(label, 'glow_animation'):
                label.glow_animation.stop()
                label.glow_animation.start()

    def start_monitoring(self):
        # Call this to start monitoring
        if self.worker and self.worker.isRunning():
            return
        
        print("UI: Starting monitoring...")
        self.reset_monitoring()
        
        self.plot_worker = PlotUpdateWorker()
        self.plot_worker.plot_data_ready.connect(self.apply_plot_data, Qt.ConnectionType.QueuedConnection)
        self.plot_worker.start()
        
        self.worker = ProcessingWorker(logic_function=self.logic_function)
        self.worker.new_frame.connect(self.update_video_frame, Qt.ConnectionType.QueuedConnection)
        self.worker.new_metrics.connect(self.process_new_data_point, Qt.ConnectionType.QueuedConnection)
        self.worker.new_metrics.connect(self.plot_worker.process_data_point, Qt.ConnectionType.QueuedConnection)
        
        self.worker.start()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.update_status(status="acquiring")

    def stop_monitoring(self):
        # Call this to stop monitoring
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.worker = None
            
        if self.plot_worker and self.plot_worker.isRunning():
            self.plot_worker.stop()
            self.plot_worker = None
            
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.update_status(status="paused")

    def reset_monitoring(self):
        # Call this to reset everything
        self.stop_monitoring()
        self.start_time = None
        
        # Reset plot worker data if it exists
        if self.plot_worker:
            self.plot_worker.reset_data()
        
        # Clear plot curves
        self.rppg_signal_curve.setData([], [])
        self.hr_curve.setData([], [])
        self.br_curve.setData([], [])
        self.sdnn_curve.setData([], [])
        self.rmssd_curve.setData([], [])
        
        self._reset_ui_values()
        self.video_display_label.setText("Press Start to Begin Monitoring")
        self.video_display_label.setPixmap(QPixmap())

    def _reset_ui_values(self):
        default_metrics = {
            "hr": {"value": "N/A", "unit": "bpm"},
            "br": {"value": "N/A", "unit": "brpm"},
            "stress": {"value": "N/A", "unit": ""}
        }
        self.update_metrics(default_metrics)
        
        self.sdnn_info_value_label.setText("N/A")
        self.rmssd_info_value_label.setText("N/A")
        self.fps_info_value_label.setText("N/A")
        self.yaw_info_value_label.setText("N/A")
        self.pitch_info_value_label.setText("N/A")
        self.roll_info_value_label.setText("N/A")
        self.blink_info_value_label.setText("N/A")
        self.blink_count_info_value_label.setText("N/A")
        self.attention_info_value_label.setText("N/A")
        self.gaze_info_value_label.setText("N/A")
        self.gaze_status_info_value_label.setText("N/A")

        self.stress_value_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY};")
        self.stress_value_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.avg_hr_label.setText("N/A")
        self.avg_br_label.setText("N/A")
        self.update_status(status="ready")
        
    def _create_metrics_group(self, title, key_prefix, icon=None):
        group_box = QGroupBox()
        main_v_layout = QVBoxLayout(group_box)
        main_v_layout.setContentsMargins(8, 6, 8, 8)
        main_v_layout.setSpacing(3)

        title_bar_layout = QHBoxLayout()
        if icon:
            icon_label = QLabel()
            icon_label.setPixmap(icon.pixmap(16, 16))
            title_bar_layout.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {COLOR_ACCENT_PRIMARY}; background-color: transparent;")
        title_bar_layout.addWidget(title_label)
        title_bar_layout.addStretch()

        content_layout = QHBoxLayout()
        value_label = QLabel("N/A")
        value_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        value_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY};")
        value_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        
        glow_colors = {
            "hr": COLOR_ACCENT_ALERT,
            "br": COLOR_ACCENT_PRIMARY,
            "stress": COLOR_ACCENT_WARN,
        }
        self._setup_glow_effect(value_label, glow_colors.get(key_prefix, COLOR_ACCENT_PRIMARY))

        unit_label = QLabel("")
        unit_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Normal))
        unit_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY};")
        unit_label.setAlignment(Qt.AlignmentFlag.AlignBottom)
        content_layout.addStretch()
        content_layout.addWidget(value_label)
        content_layout.addWidget(unit_label)
        content_layout.addStretch()
        main_v_layout.addLayout(title_bar_layout)
        main_v_layout.addLayout(content_layout)
        self._apply_shadow_effect(group_box)
        setattr(self, f"{key_prefix}_value_label", value_label)
        setattr(self, f"{key_prefix}_unit_label", unit_label)
        return group_box

    def _create_plot(self, title, ylabel, pen=None, y_range=None):
        plot_widget = PlotWidget(title=title)
        plot_widget.setLabel('left', ylabel)
        plot_widget.setLabel('bottom', "Time (s)")
        plot_widget.showGrid(x=True, y=True, alpha=0.2)
        plot_widget.getAxis('left').setTextPen(COLOR_TEXT_SECONDARY)
        plot_widget.getAxis('bottom').setTextPen(COLOR_TEXT_SECONDARY)
        plot_widget.getAxis('left').setWidth(50)

        if y_range:
            plot_widget.setYRange(y_range[0], y_range[1], padding=0)

        if pen: plot_widget.plot(pen=pen)
        self._apply_shadow_effect(plot_widget)
        return plot_widget

    @pyqtSlot(np.ndarray)
    def update_video_frame(self, cv_image):
        if cv_image is None:
            self.video_display_label.setText("No Video Signal")
            return
        h, w, ch = cv_image.shape
        qt_image = QImage(cv_image.data, w, h, ch * w, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_display_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_display_label.setPixmap(scaled_pixmap)
        
    @pyqtSlot(dict)
    def process_new_data_point(self, data_point):
        # updates UI and status
        self.update_metrics(data_point)
        
        hr_val = data_point.get("hr", {}).get("value")
        br_val = data_point.get("br", {}).get("value")
        self.update_status("monitoring", avg_hr=hr_val, avg_br=br_val)
        
        timestamp = data_point.get("timestamp")
        if timestamp is not None and self.start_time is None:
            self.start_time = timestamp
        
    @pyqtSlot(dict)
    def update_metrics(self, metrics_data):
        # Data is handled here 
        # Updates all metric labels and triggers glow for changed values
        metric_map = {
            "hr": (self.hr_value_label, self.hr_unit_label),
            "br": (self.br_value_label, self.br_unit_label),
            "stress": (self.stress_value_label, self.stress_unit_label),
        }
        for key, data in metrics_data.items():
            if key in metric_map:
                value_label, unit_label = metric_map[key]
                value = data.get("value")
                if value is None or (isinstance(value, (int, float)) and value == 0 and key != 'stress'):
                    continue
                
                current_text = f"{value:.1f}" if isinstance(value, float) else str(value)
                self._update_label_and_glow(value_label, current_text)
                unit_label.setText(data.get("unit", ""))
    
                if key == "stress":
                    # Change color for stress level
                    stress_color = {"Low Intensity": COLOR_ACCENT_SECONDARY, "Medium Intensity": COLOR_ACCENT_WARN, "High Intensity": COLOR_ACCENT_ALERT}.get(current_text, COLOR_TEXT_PRIMARY)
                    value_label.setStyleSheet(f"color: {stress_color};")
                    value_label.setFont(QFont("Segoe UI", 20 if current_text != "N/A" else 24, QFont.Weight.Bold))
    
        # HRV metrics
        sdnn_data = metrics_data.get("sdnn")
        if sdnn_data and sdnn_data.get("value") is not None and sdnn_data.get("value") != 0:
            self._update_label_and_glow(self.sdnn_info_value_label, f"{sdnn_data.get('value'):.1f} {sdnn_data.get('unit','')}")
        
        rmssd_data = metrics_data.get("rmssd")
        if rmssd_data and rmssd_data.get("value") is not None and rmssd_data.get("value") != 0:
            self._update_label_and_glow(self.rmssd_info_value_label, f"{rmssd_data.get('value'):.1f} {rmssd_data.get('unit','')}")
    
        # Signal/pose metrics
        if metrics_data.get("fps", {}).get("value") is not None:
            self._update_label_and_glow(self.fps_info_value_label, f"{metrics_data['fps']['value']:.1f}")
    
        if metrics_data.get("quality_score", {}).get("value") is not None:
            value = metrics_data["quality_score"]["value"]
            self._update_label_and_glow(self.quality_score_value_label, f"{value}")
    
        if metrics_data.get("yaw", {}).get("value") is not None:
            self._update_label_and_glow(self.yaw_info_value_label, f"{metrics_data['yaw']['value']:.0f}°")
        if metrics_data.get("pitch", {}).get("value") is not None:
            self._update_label_and_glow(self.pitch_info_value_label, f"{metrics_data['pitch']['value']:.0f}°")
        if metrics_data.get("roll", {}).get("value") is not None:
            self._update_label_and_glow(self.roll_info_value_label, f"{metrics_data['roll']['value']:.0f}°")
    
        # Cognitive metrics
        cognitive_data = metrics_data.get("cognitive")
        if cognitive_data:
            blinks = cognitive_data.get("blinks", {})
            blink_rate = blinks.get("bpm", 0)
            if isinstance(blink_rate, float):
                blink_rate_str = f"{blink_rate:.1f}"
            else:
                blink_rate_str = str(blink_rate)
            self._update_label_and_glow(self.blink_info_value_label, blink_rate_str)
            blink_count = blinks.get("count", None)
            if blink_count is not None:
                self._update_label_and_glow(self.blink_count_info_value_label, blink_count)

            attention = cognitive_data.get("attention", None)
            if attention is not None:
                try:
                    attention_val = float(attention)
                    attention_str = f"{attention_val:.1f}"
                except (ValueError, TypeError):
                    attention_str = str(attention)
                self._update_label_and_glow(self.attention_info_value_label, attention_str)

            gaze = cognitive_data.get("gaze", None)
            if gaze and isinstance(gaze, (list, tuple)) and len(gaze) > 2:
                gaze_str = f"{gaze[0]:.2f}, {gaze[1]:.2f}, {gaze[2]:.2f}"
                self._update_label_and_glow(self.gaze_info_value_label, gaze_str)

            gaze_state = cognitive_data.get("gaze_state", None)
            if gaze_state is not None:
                self._update_label_and_glow(self.gaze_status_info_value_label, gaze_state)

    @pyqtSlot(str, object, object)
    def update_status(self, status="monitoring", avg_hr=None, avg_br=None):
        status_map = {"monitoring": (self.icon_status_ok, "Monitoring", COLOR_ACCENT_SECONDARY), "acquiring": (self.icon_status_search, "Acquiring Signal...", COLOR_ACCENT_PRIMARY), "paused": (self.icon_status_warn, "Paused", COLOR_ACCENT_WARN), "ready": (self.icon_status_ready, "Ready", COLOR_TEXT_SECONDARY)}
        icon, text, color = status_map.get(status.lower(), status_map["ready"])
        self.status_icon_label.setPixmap(icon.pixmap(24, 24))
        self.status_text_label.setText(text)
        self.status_text_label.setStyleSheet(f"color: {color};")
        if avg_hr is not None: self.avg_hr_label.setText(f"{avg_hr:.1f} bpm")
        if avg_br is not None: self.avg_br_label.setText(f"{avg_br:.1f} brpm")

    @pyqtSlot(dict)
    def apply_plot_data(self, plot_data):
        # Receives new plot data from worker and updates all plots
        try:
            rppg_times, rppg_values = plot_data.get('rppg_signal_data', (np.array([]), np.array([])))
            hr_times, hr_values = plot_data['hr_data']
            br_times, br_values = plot_data['br_data']
            sdnn_times, sdnn_values = plot_data['sdnn_data']
            rmssd_times, rmssd_values = plot_data['rmssd_data']
            current_elapsed_time = plot_data['current_time']
    
            self.rppg_signal_curve.setData(rppg_times, rppg_values)
            if rppg_times.size > 1:
                self.rppg_plot_widget.setXRange(rppg_times[0], rppg_times[-1])
            
            self.hr_curve.setData(hr_times, hr_values)
            self.br_curve.setData(br_times, br_values)
            self.sdnn_curve.setData(sdnn_times, sdnn_values)
            self.rmssd_curve.setData(rmssd_times, rmssd_values)
    
            if hr_times.size > 0:
                # Keep time window for plots at last 30 seconds
                view_end_time = current_elapsed_time + 1
                view_start_time = max(0, view_end_time - 30)
    
                for plot in [self.hr_plot_widget, self.br_plot_widget, self.hrv_plot_widget]:
                    plot.getPlotItem().setXRange(view_start_time, view_end_time, padding=0)
                    
        except Exception as e:
            print(f"Error applying plot data: {e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def closeEvent(self, event):
        print("Closing application - stopping all workers")
        self.stop_monitoring()
        event.accept()

if __name__ == '__main__':
    # Can run this file to test the UI without the full logic
    # Make sure to change "from UI.ui_worker import ProcessingWorker" to "from ui_worker import ProcessingWorker"
    def dummy_logic(*args, **kwargs):
        print("Dummy logic is running. Press Stop to end.")
        while not kwargs['should_stop']():
            time.sleep(1)
            
    app = QApplication(sys.argv)
    main_window = AppWindow(logic_function=dummy_logic)
    main_window.show()
    sys.exit(app.exec())
