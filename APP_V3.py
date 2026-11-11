# -*- coding: utf-8 -*-
"""
AutoPlate - Plaka tanıma veri seti hazırlama stüdyosu
=====================================================
1) Görsel İndirici  2) YOLO Etiketleme  3) Hatalı Etiket Silici
4) Ad Değiştirici   5) Train/Val/Test Ayırıcı

Yenilikler: Sidebar tabanlı arayüz, Aydınlık/Karanlık tema, animasyonlar,
çökme koruması, otomatik kayıt ve kaldığı yerden devam etme.
"""
from __future__ import annotations

import sys
import os
import re
import json
import time
import uuid
import random
import shutil
import threading
import traceback
import html as _h
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QTextEdit,
    QFileDialog, QMessageBox, QFrame, QComboBox, QSplitter, QScrollArea,
    QStackedWidget, QProgressBar, QButtonGroup, QGraphicsOpacityEffect,
    QGraphicsDropShadowEffect, QDialog, QSizePolicy, QAbstractSpinBox,
)
from PySide6.QtCore import (
    Qt, QObject, QThread, Signal, QMutex, QWaitCondition, QPropertyAnimation,
    QEasingCurve, QVariantAnimation, QTimer, QRect, QPoint, QByteArray,
    QParallelAnimationGroup, QAbstractAnimation, QEvent,
)
from PySide6.QtGui import (
    QFont, QTextCursor, QIcon, QColor, QPainter, QPixmap, QPen, QKeySequence,
    QShortcut, QPolygonF,
)
from PySide6.QtCore import QPointF

# --- Opsiyonel kütüphaneler (Hızlı açılış için Lazy Import ile adım içinde yüklenir) ---
Camoufox = None
PlaywrightError = Exception
torch = None
torchvision = None
cv2 = None
YOLO = None

APP_NAME = "AutoPlate"
APP_VERSION = "1.0.0"


# =============================================================================
# Yardımcılar: dosya yolları, atomik JSON, kaynak bulma
# =============================================================================
def app_data_dir() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", str(Path.home())))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    p = root / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def resource_path(name: str) -> str:
    candidates = []
    if getattr(sys, "_MEIPASS", None):
        candidates.append(Path(sys._MEIPASS) / name)
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).parent / name)
    candidates.append(Path(__file__).resolve().parent / name)
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


def atomic_write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_crash_log(text: str) -> None:
    try:
        with open(app_data_dir() / "crash.log", "a", encoding="utf-8") as f:
            f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} =====\n{text}\n")
    except OSError:
        pass


# =============================================================================
# Oturum (kayıt) yöneticisi - kaldığı yerden devam için
# =============================================================================
class SessionStore:
    def __init__(self, path: Path):
        self.path = path
        self.dir = path.parent
        self.lock = threading.RLock()
        self._last_save = 0.0
        self.data = self._load()

    @staticmethod
    def _blank():
        return {
            "version": 1, "config": {},
            "pipeline": {"steps": [], "completed": [], "current": None, "status": "idle"},
            "progress": {}, "updated": "",
        }

    def _load(self):
        blank = self._blank()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            for k, v in blank.items():
                d.setdefault(k, v)
            return d
        except (OSError, json.JSONDecodeError):
            return blank

    def save(self, force=True):
        with self.lock:
            now = time.time()
            if not force and now - self._last_save < 1.0:
                return
            self._last_save = now
            self.data["updated"] = datetime.now().isoformat(timespec="seconds")
            try:
                atomic_write_json(self.path, self.data)
            except OSError:
                pass

    def set_config(self, cfg):
        with self.lock:
            self.data["config"] = cfg
        self.save()

    def begin(self, steps, resume):
        with self.lock:
            p = self.data["pipeline"]
            if not resume:
                p["steps"] = list(steps)
                p["completed"] = []
                for s in steps:
                    self.data["progress"].pop(str(s), None)
            p["status"] = "running"
            p["current"] = None
        self.save()

    def set_current(self, step):
        with self.lock:
            self.data["pipeline"]["current"] = step
        self.save()

    def mark_completed(self, step):
        with self.lock:
            p = self.data["pipeline"]
            if step not in p["completed"]:
                p["completed"].append(step)
            p["current"] = None
        self.save()

    def finish(self, status):
        with self.lock:
            self.data["pipeline"]["status"] = status
            if status == "done":
                self.data["progress"] = {}
                self.data["pipeline"]["current"] = None
        self.save()

    def clear(self):
        with self.lock:
            cfg = self.data.get("config", {})
            self.data = self._blank()
            self.data["config"] = cfg
        self.save()

    def pending_steps(self):
        with self.lock:
            p = self.data["pipeline"]
            if p["status"] in ("running", "interrupted", "error"):
                return [s for s in p["steps"] if s not in p["completed"]]
        return []

    def completed_steps(self):
        with self.lock:
            return list(self.data["pipeline"]["completed"])

    def get_progress(self, step):
        with self.lock:
            return dict(self.data["progress"].get(str(step), {}))

    def set_progress(self, step, force=False, **kw):
        with self.lock:
            self.data["progress"].setdefault(str(step), {}).update(kw)
        self.save(force=force)


# =============================================================================
# Temalar
# =============================================================================
THEMES = {
    "dark": dict(
        bg="#0A0C12", sidebar="#0D1017", sidebar_hover="#161B26",
        surface="#11151D", surface2="#171C27", hover="#1F2533",
        border="#232A38", border_hover="#3A4764",
        text="#E8ECF4", muted="#8B95A9",
        accent="#7C83FF", accent_hover="#9399FF", accent_press="#6A71F0",
        accent_soft="rgba(124,131,255,16%)", on_accent="#FFFFFF",
        success="#34D399", warn="#FBBF24", danger="#F87171",
        input="#0E121A", console="#070A10", console_text="#C9D4E8",
    ),
    "light": dict(
        bg="#F4F6FB", sidebar="#FFFFFF", sidebar_hover="#F1F3F9",
        surface="#FFFFFF", surface2="#F3F5FA", hover="#E8ECF5",
        border="#E2E6EF", border_hover="#C2CADB",
        text="#131826", muted="#677085",
        accent="#4F46E5", accent_hover="#4338CA", accent_press="#3730A3",
        accent_soft="rgba(79,70,229,10%)", on_accent="#FFFFFF",
        success="#16A34A", warn="#D97706", danger="#DC2626",
        input="#FFFFFF", console="#0F1420", console_text="#D5DDEE",
    ),
}
PAL = dict(THEMES["dark"])

QSS = """
QMainWindow, QWidget#root { background: @bg; }
QWidget { color: @text; }
QLabel { background: transparent; }
QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }
QToolTip { background: @surface; color: @text; border: 1px solid @border; padding: 6px 8px; border-radius: 6px; }

/* Sidebar */
QFrame#sidebar { background: @sidebar; border-right: 1px solid @border; }
QLabel#brandName { font-size: 18px; font-weight: 700; }
QLabel#brandSub, QLabel#navSection, QLabel#footerLbl { color: @muted; font-size: 11px; font-weight: 600; }
QPushButton#navBtn {
    background: transparent; border: none; border-radius: 10px; text-align: left;
    padding: 0 34px 0 18px; color: @muted; font-weight: 600; font-size: 13px;
}
QPushButton#navBtn:hover { background: @sidebar_hover; color: @text; }
QPushButton#navBtn:checked { background: @accent_soft; color: @accent; }
QFrame#navIndicator { background: @accent; border-radius: 1px; border: none; }

/* Header */
QLabel#pageTitle { font-size: 24px; font-weight: 700; }
QLabel#pageSub { color: @muted; font-size: 13px; }

/* Cards */
QFrame#card { background: @surface; border: 1px solid @border; border-radius: 14px; }
QLabel#cardTitle { font-size: 15px; font-weight: 700; }
QLabel#cardSub, QLabel#hint { color: @muted; }
QLabel#fieldLabel { color: @muted; font-size: 12px; font-weight: 600; }
QLabel#hint { background: @surface2; border: 1px solid @border; border-radius: 10px; padding: 12px 14px; }

/* Inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: @input; border: 1px solid @border; border-radius: 9px;
    padding: 8px 12px; color: @text; min-height: 20px;
    selection-background-color: @accent; selection-color: @on_accent;
}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover { border: 1px solid @border_hover; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid @accent; }
QSpinBox, QDoubleSpinBox { padding-right: 26px; }
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right; width: 22px; border: none; background: transparent; margin-top: 3px; }
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right; width: 22px; border: none; background: transparent; margin-bottom: 3px; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url("@arrow_up"); width: 10px; height: 10px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url("@arrow_down"); width: 10px; height: 10px; }
QComboBox::drop-down { border: none; width: 28px; subcontrol-origin: padding; subcontrol-position: center right; }
QComboBox::down-arrow { image: url("@arrow_down"); width: 10px; height: 10px; }
QComboBox QAbstractItemView {
    background: @surface; color: @text; border: 1px solid @border; border-radius: 8px;
    padding: 4px; selection-background-color: @accent_soft; selection-color: @accent; outline: none; }

/* Buttons */
QPushButton {
    background: @surface2; border: 1px solid @border; border-radius: 9px;
    padding: 8px 16px; color: @text; font-weight: 600;
}
QPushButton:hover { background: @hover; border-color: @border_hover; }
QPushButton:pressed { background: @surface; }
QPushButton:disabled { color: @muted; background: @surface; border-color: @border; }
QPushButton#primary { background: @accent; color: @on_accent; border: none; padding: 12px 18px; font-size: 14px; }
QPushButton#primary:hover { background: @accent_hover; }
QPushButton#primary:pressed { background: @accent_press; }
QPushButton#primary:disabled { background: @surface2; color: @muted; }
QPushButton#success { background: @success; color: #08130D; border: none; }
QPushButton#success:disabled { background: @surface2; color: @muted; }
QPushButton#danger { background: transparent; color: @danger; border: 1px solid @danger; }
QPushButton#danger:hover { background: @danger; color: #FFFFFF; }
QPushButton#danger:disabled { color: @muted; border-color: @border; background: transparent; }
QPushButton#warn { background: @warn; color: #1A1203; border: none; }
QPushButton#warn:disabled { background: @surface2; color: @muted; }
QPushButton#ghost { background: transparent; border: none; color: @muted; padding: 6px 10px; }
QPushButton#ghost:hover { background: @hover; color: @text; }
QPushButton#iconBtn { padding: 0; min-width: 38px; max-width: 38px; min-height: 38px; max-height: 38px; font-size: 16px; }

/* Run bar */
QFrame#runBar { background: @surface; border: 1px solid @border; border-radius: 12px; }
QLabel#statusLbl { font-weight: 600; }
QLabel#elapsedLbl { color: @muted; font-family: "Cascadia Mono", Consolas, monospace; }
QProgressBar { background: @surface2; border: none; border-radius: 4px; max-height: 8px; min-height: 8px; }
QProgressBar::chunk { background: @accent; border-radius: 4px; }

/* Step chips */
QFrame#stepChip { background: @surface2; border: 1px solid @border; border-radius: 12px; }
QFrame#stepChip:hover { border: 1px solid @border_hover; }
QFrame#stepChip[state="running"] { border: 1px solid @accent; background: @accent_soft; }
QFrame#stepChip[state="running"][pulse="true"] { border: 1px solid @accent_hover; }
QFrame#stepChip[state="done"] { border: 1px solid @success; }
QFrame#stepChip[state="error"] { border: 1px solid @danger; }
QFrame#stepChip[state="paused"] { border: 1px solid @warn; }
QLabel#chipNum { background: @surface; border: 1px solid @border; border-radius: 14px; font-weight: 700; color: @muted; }
QFrame#stepChip[state="running"] QLabel#chipNum { background: @accent; color: @on_accent; border: none; }
QFrame#stepChip[state="done"] QLabel#chipNum { background: @success; color: #08130D; border: none; }
QFrame#stepChip[state="error"] QLabel#chipNum { background: @danger; color: #FFFFFF; border: none; }
QFrame#stepChip[state="paused"] QLabel#chipNum { background: @warn; color: #1A1203; border: none; }
QLabel#chipTitle { font-weight: 700; }
QLabel#chipState { color: @muted; font-size: 11px; }

/* Banner, toast */
QFrame#banner { background: @accent_soft; border: 1px solid @accent; border-radius: 12px; }
QLabel#bannerTitle { font-weight: 700; font-size: 14px; }
QFrame#toast { background: @surface; border: 1px solid @border; border-radius: 12px; border-left: 4px solid @accent; }
QFrame#toast[kind="error"] { border-left: 4px solid @danger; }
QFrame#toast[kind="success"] { border-left: 4px solid @success; }
QFrame#toast[kind="warn"] { border-left: 4px solid @warn; }

/* Console */
QTextEdit#console {
    background: @console; color: @console_text; border: 1px solid @border; border-radius: 12px;
    padding: 10px; selection-background-color: #3B4A6B; }
QFrame#consoleBar { background: @surface; border: 1px solid @border; border-radius: 10px; }
QLabel#consolePreview { color: @muted; font-family: "Cascadia Mono", Consolas, monospace; font-size: 12px; }
QLabel#consoleTitle { font-weight: 700; }
QSplitter::handle:vertical { background: @border; margin: 4px 120px; height: 2px; border-radius: 1px; }
QSplitter::handle:vertical:hover { background: @accent; }

/* Scrollbars */
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: @border_hover; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: @muted; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: @border_hover; border-radius: 4px; min-width: 30px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* Dialogs */
QDialog, QMessageBox { background: @surface; }
"""


def build_qss(theme: dict, arrow_up: str, arrow_down: str) -> str:
    tokens = dict(theme)
    tokens["arrow_up"] = arrow_up
    tokens["arrow_down"] = arrow_down
    out = QSS
    for k in sorted(tokens, key=len, reverse=True):
        out = out.replace("@" + k, tokens[k])
    return out


def build_arrow_assets(name: str, color_hex: str):
    d = app_data_dir() / "assets"
    d.mkdir(exist_ok=True)
    paths = {}
    for kind in ("up", "down"):
        pm = QPixmap(12, 12)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(color_hex), 1.8)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        if kind == "up":
            p.drawPolyline(QPolygonF([QPointF(2.5, 7.5), QPointF(6, 4), QPointF(9.5, 7.5)]))
        else:
            p.drawPolyline(QPolygonF([QPointF(2.5, 4.5), QPointF(6, 8), QPointF(9.5, 4.5)]))
        p.end()
        fp = d / f"{kind}_{name}.png"
        pm.save(str(fp))
        paths[kind] = fp.as_posix()
    return paths["up"], paths["down"]


# =============================================================================
# Dil sözlüğü
# =============================================================================
LANG = {
    "tr": {
        "app_sub": "",
        "sec_flow": "İŞ AKIŞI", "sec_sys": "SİSTEM",
        "nav0": "Genel Bakış", "nav1": "1 · Görsel İndirici", "nav2": "2 · YOLO Etiketleme",
        "nav3": "3 · Hatalı Etiket Silici", "nav4": "4 · Ad Değiştirici",
        "nav5": "5 · Dataset Ayırıcı", "nav6": "Ayarlar",
        "t0": "Genel Bakış", "sub0": "Çalışma alanını belirleyin ve tüm hattı tek tuşla çalıştırın.",
        "t1": "Görsel İndirici", "sub1": "Galeriden araç görsellerini ve plaka notlarını indirir.",
        "t2": "YOLO Etiketleme", "sub2": "Ham görselleri YOLO modeliyle etiketler ve çizimli kopyalarını üretir.",
        "t3": "Hatalı Etiket Silici", "sub3": "Boş veya çoklu etiketli örnekleri tüm klasörlerden temizler.",
        "t4": "Ad Değiştirici", "sub4": "Dosyaları çakışmadan img(N) biçiminde yeniden adlandırır.",
        "t5": "Dataset Ayırıcı", "sub5": "Veriyi Train / Val / Test olarak YOLO klasör yapısına böler.",
        "t6": "Ayarlar", "sub6": "Görünüm, dil, kayıt ve yapılandırma tercihleri.",
        "chip1": "İndir", "chip2": "Etiketle", "chip3": "Temizle", "chip4": "Adlandır", "chip5": "Böl",
        "st_idle": "Bekliyor", "st_running": "Çalışıyor", "st_done": "Tamamlandı",
        "st_error": "Hata", "st_paused": "Durduruldu",
        "ws_title": "Çalışma Alanı", "ws_sub": "1–4. adımlar bu klasörleri ortak kullanır.",
        "base_dir_lbl": "Ana Çalışma Dizini", "browse_btn": "Klasör Seç",
        "raw_folder_lbl": "Ham Görsel Klasörü", "drawn_folder_lbl": "Çizimli Görsel Klasörü",
        "labels_folder_lbl": "TXT Etiket Klasörü",
        "pl_title": "Otomatik Hat", "pl_sub": "Seçili ayarlarla 1 → 5 adımlarını sırayla çalıştırır.",
        "run_all_btn": "🚀  Tümünü Sırayla Çalıştır",
        "stop_btn": "⏹  Durdur", "continue_btn": "▶  Devam Et / Onayla",
        "status_ready": "Hazır", "status_wait": "Kullanıcı onayı bekleniyor…",
        "resume_title": "Yarım kalan işlem bulundu",
        "resume_text": "Uygulama beklenmedik biçimde kapanmış veya durdurulmuş. Kalan adımlar: {steps}",
        "resume_btn": "Kaldığı Yerden Devam Et", "discard_btn": "Yoksay",
        "log_group": "KONSOL", "clear_log": "Temizle", "copy_log": "Kopyala",
        "tip_toggle": "Konsolu aç/kapat (Ctrl+L)", "tip_max": "Konsolu büyüt/küçült",
        "tip_theme": "Tema değiştir (Ctrl+T)",
        "s1_url_lbl": "Hedef Galeri URL", "s1_start_page": "Başlangıç Sayfası",
        "s1_page_count": "Taranacak Sayfa Sayısı", "s1_wait_time": "Bekleme Süresi (sn)",
        "s1_notes_file": "Plaka Not Dosyası (.txt)", "s1_state_file": "Durum Dosyası (.json)",
        "s1_link_sel": "Link Seçici (CSS)", "s1_img_sel": "Görsel Seçici (CSS)",
        "s1_btn": "▶  Görsel İndiriciyi Çalıştır",
        "s2_model_lbl": "YOLO Model Dosyası (.pt)", "s2_model_btn": "Model Seç",
        "s2_class_lbl": "Atanacak Class ID", "s2_conf_lbl": "Güven Eşiği (Confidence)",
        "s2_imgsz_lbl": "Görsel Boyutu (imgsz)", "s2_btn": "▶  YOLO Etiketlemeyi Çalıştır",
        "s3_info": ("Bu adım Ham Görsel, Çizimli Görsel ve TXT Etiket klasörlerini tarar.\n"
                    "• Etiketi boş olan (0 tespit) veya hiç bulunmayan görseller,\n"
                    "• 2 veya daha fazla etiketi (çoklu tespit) olan görseller\n"
                    "her 3 klasörden de kalıcı olarak silinir."),
        "s3_btn": "▶  Hatalı Etiket Silicisini Çalıştır",
        "s4_start_no": "Başlangıç Numarası (img(1) için 1)", "s4_exclude": "Hariç Tutulacak Dosyalar (virgülle)",
        "s4_info": ("Yalnızca Ham Görsel Klasöründe kalan (3. adımdan sağ çıkan) dosyalar baz alınır.\n"
                    "• Görseller ve etiketler çakışma olmadan img(N) biçiminde yeniden sıralanır.\n"
                    "• 1. adımdaki Plaka Not Dosyası eşzamanlı güncellenir, silinenlerin plakaları temizlenir."),
        "s4_btn": "▶  Ad Değiştiriciyi Çalıştır",
        "s5_out_dir": "Çıktı Dataset Klasör Adı", "s5_train_ratio": "Train Oranı", "s5_val_ratio": "Val Oranı",
        "s5_test": "Test Oranı: %{p}",
        "s5_src1_img": "1. Kaynak Resim Klasörü", "s5_src1_txt": "1. Kaynak TXT Klasörü",
        "s5_src2_img": "2. Kaynak Resim (opsiyonel)", "s5_src2_txt": "2. Kaynak TXT (opsiyonel)",
        "s5_btn": "▶  Dataset Ayırıcıyı Çalıştır",
        "set_appearance": "Görünüm", "set_behavior": "Davranış", "set_data": "Yapılandırma ve Veri",
        "set_lang_lbl": "Uygulama Dili", "set_theme_lbl": "Tema",
        "theme_dark": "Karanlık", "theme_light": "Aydınlık",
        "set_font_lbl": "Konsol Yazı Boyutu (pt)", "set_anim_lbl": "Arayüz Animasyonları",
        "anim_on": "Açık", "anim_off": "Kapalı",
        "set_confirm_lbl": "Silme işleminden önce onay iste", "yes": "Evet", "no": "Hayır",
        "set_save_btn": "💾  Ayarları JSON Olarak Kaydet", "set_load_btn": "📂  Ayarları JSON'dan Yükle",
        "set_open_data": "📁  Kayıt Klasörünü Aç", "set_reset_session": "🧹  Kayıtlı Oturumu Sıfırla",
        "autosave": "Otomatik kayıt: {t}",
        "msg_busy": "Şu anda çalışan bir işlem var.",
        "msg_bad_dir": "Ana çalışma dizini bulunamadı: {p}",
        "msg_bad_ratio": "Train + Val oranı 1.0'dan küçük olmalıdır.",
        "msg_confirm_del": "3. adım, hatalı etiketli görselleri KALICI olarak siler. Devam edilsin mi?",
        "msg_confirm_title": "Onay",
        "msg_done": "Tüm seçili işlemler başarıyla tamamlandı.",
        "msg_stopped": "İşlem durduruldu. İlerleme kaydedildi, daha sonra devam edebilirsiniz.",
        "msg_error": "Bir hata oluştu. İlerleme kaydedildi.",
        "msg_saved": "Ayarlar kaydedildi.", "msg_loaded": "Ayarlar yüklendi.",
        "msg_reset": "Kayıtlı oturum sıfırlandı.",
        "msg_close_running": "Bir işlem çalışıyor. Durdurup kapatılsın mı? (İlerleme kaydedilir)",
        "err_title": "Beklenmeyen bir hata oluştu",
        "err_sub": "Uygulama çökmedi. İlerlemeniz kaydedildi ve kaldığınız yerden devam edebilirsiniz.",
        "err_copy": "Hatayı Kopyala", "err_close": "Kapat", "err_logdir": "Kayıt Klasörü",
        "dlg_model": "YOLO Modeli Seç", "dlg_dir": "Klasör Seç",
        "ph_s5_img": "Örn: motorlar", "ph_s5_txt": "Örn: etiketler_moto",
        "pending_toast": "Yarım kalan işlem bulundu. Genel Bakış'tan devam edebilirsiniz.",
    },
    "en": {
        "app_sub": "",
        "sec_flow": "WORKFLOW", "sec_sys": "SYSTEM",
        "nav0": "Overview", "nav1": "1 · Image Downloader", "nav2": "2 · YOLO Labeling",
        "nav3": "3 · Bad Label Remover", "nav4": "4 · Batch Renamer",
        "nav5": "5 · Dataset Splitter", "nav6": "Settings",
        "t0": "Overview", "sub0": "Set your workspace and run the entire pipeline in one click.",
        "t1": "Image Downloader", "sub1": "Downloads vehicle images and plate notes from the gallery.",
        "t2": "YOLO Labeling", "sub2": "Labels raw images with a YOLO model and renders annotated copies.",
        "t3": "Bad Label Remover", "sub3": "Removes empty or multi-label samples from all folders.",
        "t4": "Batch Renamer", "sub4": "Safely renames files to img(N) without collisions.",
        "t5": "Dataset Splitter", "sub5": "Splits data into Train / Val / Test using the YOLO folder layout.",
        "t6": "Settings", "sub6": "Appearance, language, autosave and configuration preferences.",
        "chip1": "Download", "chip2": "Label", "chip3": "Clean", "chip4": "Rename", "chip5": "Split",
        "st_idle": "Idle", "st_running": "Running", "st_done": "Done",
        "st_error": "Error", "st_paused": "Stopped",
        "ws_title": "Workspace", "ws_sub": "Steps 1–4 share these folders.",
        "base_dir_lbl": "Base Directory", "browse_btn": "Browse",
        "raw_folder_lbl": "Raw Image Folder", "drawn_folder_lbl": "Annotated Image Folder",
        "labels_folder_lbl": "TXT Labels Folder",
        "pl_title": "Automated Pipeline", "pl_sub": "Runs steps 1 → 5 sequentially with the current settings.",
        "run_all_btn": "🚀  Run All Sequentially",
        "stop_btn": "⏹  Stop", "continue_btn": "▶  Continue / Confirm",
        "status_ready": "Ready", "status_wait": "Waiting for user confirmation…",
        "resume_title": "Unfinished session found",
        "resume_text": "The app was closed unexpectedly or stopped. Remaining steps: {steps}",
        "resume_btn": "Resume Where It Left Off", "discard_btn": "Dismiss",
        "log_group": "CONSOLE", "clear_log": "Clear", "copy_log": "Copy",
        "tip_toggle": "Toggle console (Ctrl+L)", "tip_max": "Enlarge / shrink console",
        "tip_theme": "Switch theme (Ctrl+T)",
        "s1_url_lbl": "Target Gallery URL", "s1_start_page": "Start Page",
        "s1_page_count": "Pages to Scan", "s1_wait_time": "Wait Time (sec)",
        "s1_notes_file": "Plate Notes File (.txt)", "s1_state_file": "State File (.json)",
        "s1_link_sel": "Link Selector (CSS)", "s1_img_sel": "Image Selector (CSS)",
        "s1_btn": "▶  Run Image Downloader",
        "s2_model_lbl": "YOLO Model File (.pt)", "s2_model_btn": "Select Model",
        "s2_class_lbl": "Target Class ID", "s2_conf_lbl": "Confidence Threshold",
        "s2_imgsz_lbl": "Image Size (imgsz)", "s2_btn": "▶  Run YOLO Labeling",
        "s3_info": ("This step scans the Raw, Annotated and TXT Label folders.\n"
                    "• Images with empty or missing label files (0 detections),\n"
                    "• Images with 2 or more labels (multiple detections)\n"
                    "are permanently deleted from all 3 folders."),
        "s3_btn": "▶  Run Bad Label Remover",
        "s4_start_no": "Start Number (1 for img(1))", "s4_exclude": "Exclude Files (comma-separated)",
        "s4_info": ("Only the surviving files in the Raw Image Folder (after step 3) are used as reference.\n"
                    "• Images and labels are renamed to img(N) safely using temporary names.\n"
                    "• The Plate Notes File from step 1 is updated and entries of deleted images are removed."),
        "s4_btn": "▶  Run Batch Renamer",
        "s5_out_dir": "Output Dataset Folder Name", "s5_train_ratio": "Train Ratio", "s5_val_ratio": "Val Ratio",
        "s5_test": "Test Ratio: {p}%",
        "s5_src1_img": "Primary Image Folder", "s5_src1_txt": "Primary TXT Folder",
        "s5_src2_img": "Secondary Image (optional)", "s5_src2_txt": "Secondary TXT (optional)",
        "s5_btn": "▶  Run Dataset Splitter",
        "set_appearance": "Appearance", "set_behavior": "Behavior", "set_data": "Configuration & Data",
        "set_lang_lbl": "Application Language", "set_theme_lbl": "Theme",
        "theme_dark": "Dark", "theme_light": "Light",
        "set_font_lbl": "Console Font Size (pt)", "set_anim_lbl": "UI Animations",
        "anim_on": "Enabled", "anim_off": "Disabled",
        "set_confirm_lbl": "Ask before destructive deletion", "yes": "Yes", "no": "No",
        "set_save_btn": "💾  Save Config to JSON", "set_load_btn": "📂  Load Config from JSON",
        "set_open_data": "📁  Open Data Folder", "set_reset_session": "🧹  Reset Saved Session",
        "autosave": "Autosave: {t}",
        "msg_busy": "A task is already running.",
        "msg_bad_dir": "Base directory not found: {p}",
        "msg_bad_ratio": "Train + Val ratio must be less than 1.0.",
        "msg_confirm_del": "Step 3 PERMANENTLY deletes images with bad labels. Continue?",
        "msg_confirm_title": "Confirm",
        "msg_done": "All selected tasks completed successfully.",
        "msg_stopped": "Stopped. Progress was saved, you can resume later.",
        "msg_error": "An error occurred. Progress was saved.",
        "msg_saved": "Settings saved.", "msg_loaded": "Settings loaded.",
        "msg_reset": "Saved session was reset.",
        "msg_close_running": "A task is running. Stop it and quit? (Progress is saved)",
        "err_title": "An unexpected error occurred",
        "err_sub": "The app did not crash. Your progress was saved and you can resume where you left off.",
        "err_copy": "Copy Error", "err_close": "Close", "err_logdir": "Data Folder",
        "dlg_model": "Select YOLO Model", "dlg_dir": "Select Directory",
        "ph_s5_img": "e.g. motorcycles", "ph_s5_txt": "e.g. labels_moto",
        "pending_toast": "An unfinished session was found. Resume it from Overview.",
    },
}


# =============================================================================
# Arayüz yardımcıları
# =============================================================================
def repolish(w: QWidget):
    st = w.style()
    for x in [w] + w.findChildren(QWidget):
        st.unpolish(x)
        st.polish(x)
    w.update()


def fade_in(w: QWidget, duration=260, start=0.0):
    old = getattr(w, "_fa", None)
    if old is not None:
        try:
            old.stop()
            old.deleteLater()
        except RuntimeError:
            pass
    eff = QGraphicsOpacityEffect(w)
    eff.setOpacity(start)
    w.setGraphicsEffect(eff)
    a = QPropertyAnimation(eff, b"opacity", w)
    a.setDuration(duration)
    a.setStartValue(start)
    a.setEndValue(1.0)
    a.setEasingCurve(QEasingCurve.OutCubic)

    def done():
        try:
            if w.graphicsEffect() is eff:
                w.setGraphicsEffect(None)
        except RuntimeError:
            pass

    a.finished.connect(lambda: QTimer.singleShot(0, done))
    w._fa = a
    a.start()


class GlowButton(QPushButton):
    """Üzerine gelince / dikkat çekmek istendiğinde parlayan buton."""
    _KEYS = {"primary": "accent", "success": "success", "danger": "danger", "warn": "warn"}

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self._fx = QGraphicsDropShadowEffect(self)
        self._fx.setOffset(0, 3)
        self._fx.setBlurRadius(0)
        self._fx.setColor(QColor(0, 0, 0, 0))
        self.setGraphicsEffect(self._fx)
        self._v = 0.0
        self._hover = QVariantAnimation(self)
        self._hover.setDuration(180)
        self._hover.valueChanged.connect(self._apply)
        self._pulse = QVariantAnimation(self)
        self._pulse.setDuration(1000)
        self._pulse.setKeyValueAt(0.0, 0.1)
        self._pulse.setKeyValueAt(0.5, 1.0)
        self._pulse.setKeyValueAt(1.0, 0.1)
        self._pulse.setLoopCount(-1)
        self._pulse.valueChanged.connect(self._apply)

    def _apply(self, v):
        self._v = float(v)
        key = self._KEYS.get(self.objectName(), "accent")
        c = QColor(PAL[key])
        c.setAlpha(int(150 * self._v))
        self._fx.setColor(c)
        self._fx.setBlurRadius(4 + 26 * self._v)

    def _glide(self, target):
        if self._pulse.state() == QAbstractAnimation.Running:
            return
        self._hover.stop()
        self._hover.setStartValue(self._v)
        self._hover.setEndValue(target)
        self._hover.start()

    def enterEvent(self, e):
        if self.isEnabled():
            self._glide(1.0)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._glide(0.0)
        super().leaveEvent(e)

    def start_pulse(self):
        if self._pulse.state() != QAbstractAnimation.Running:
            self._hover.stop()
            self._pulse.start()

    def stop_pulse(self):
        self._pulse.stop()
        self._apply(0.0)


class NavButton(QPushButton):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setObjectName("navBtn")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(42)
        self._state = "idle"
        self._pulse = False

    def set_state(self, s):
        self._state = s
        self.update()

    def set_pulse(self, on):
        self._pulse = on
        if self._state == "running":
            self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._state == "idle":
            return
        col = {"running": PAL["accent"], "done": PAL["success"],
               "error": PAL["danger"], "paused": PAL["warn"]}.get(self._state)
        if not col:
            return
        c = QColor(col)
        if self._state == "running" and self._pulse:
            c.setAlpha(100)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        p.drawEllipse(QPointF(self.width() - 18, self.height() / 2), 4.5, 4.5)
        p.end()


class StepChip(QFrame):
    clicked = Signal(int)

    def __init__(self, step, parent=None):
        super().__init__(parent)
        self.step = step
        self.setObjectName("stepChip")
        self.setProperty("state", "idle")
        self.setProperty("pulse", "false")
        self.setCursor(Qt.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)
        self.num = QLabel(str(step))
        self.num.setObjectName("chipNum")
        self.num.setFixedSize(28, 28)
        self.num.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.num)
        col = QVBoxLayout()
        col.setSpacing(1)
        self.title = QLabel()
        self.title.setObjectName("chipTitle")
        self.state_lbl = QLabel()
        self.state_lbl.setObjectName("chipState")
        col.addWidget(self.title)
        col.addWidget(self.state_lbl)
        lay.addLayout(col, 1)

    def set_state(self, state, text):
        self.setProperty("state", state)
        self.state_lbl.setText(text)
        self.num.setText({"done": "✓", "error": "!"}.get(state, str(self.step)))
        repolish(self)

    def set_pulse(self, on):
        if self.property("state") == "running":
            self.setProperty("pulse", "true" if on else "false")
            repolish(self)

    def mousePressEvent(self, e):
        self.clicked.emit(self.step)
        super().mousePressEvent(e)


class Toast(QFrame):
    def __init__(self, parent, text, kind):
        super().__init__(parent)
        self.setObjectName("toast")
        self.setProperty("kind", kind)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10)
        icon = {"error": "⚠", "success": "✓", "warn": "!", "info": "ℹ"}.get(kind, "ℹ")
        ic = QLabel(icon)
        ic.setStyleSheet("font-size:16px; font-weight:700;")
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setMaximumWidth(380)
        lay.addWidget(ic)
        lay.addWidget(lbl)
        self.setGraphicsEffect(None)
        self.adjustSize()


class ToastHost(QObject):
    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self.win = window
        self.toasts: list[Toast] = []

    def show(self, text, kind="info", ms=4800):
        t = Toast(self.win, text, kind)
        t.setFixedWidth(min(440, max(260, t.sizeHint().width())))
        t.adjustSize()
        self.toasts.append(t)
        t.show()
        t.raise_()
        self.reflow()
        eff = QGraphicsOpacityEffect(t)
        t.setGraphicsEffect(eff)
        end = t.pos()
        a_op = QPropertyAnimation(eff, b"opacity", t)
        a_op.setDuration(260 if self.win.animations_enabled else 1)
        a_op.setStartValue(0.0)
        a_op.setEndValue(1.0)
        a_pos = QPropertyAnimation(t, b"pos", t)
        a_pos.setDuration(320 if self.win.animations_enabled else 1)
        a_pos.setStartValue(QPoint(end.x() + 50, end.y()))
        a_pos.setEndValue(end)
        a_pos.setEasingCurve(QEasingCurve.OutCubic)
        grp = QParallelAnimationGroup(t)
        grp.addAnimation(a_op)
        grp.addAnimation(a_pos)
        grp.start()
        t._grp = grp
        QTimer.singleShot(ms, lambda: self._dismiss(t, eff))

    def _dismiss(self, t, eff):
        if t not in self.toasts:
            return
        a = QPropertyAnimation(eff, b"opacity", t)
        a.setDuration(300 if self.win.animations_enabled else 1)
        a.setStartValue(1.0)
        a.setEndValue(0.0)

        def done():
            if t in self.toasts:
                self.toasts.remove(t)
            t.deleteLater()
            self.reflow()

        a.finished.connect(done)
        t._out = a
        a.start()

    def reflow(self):
        y = 84
        for t in self.toasts:
            t.move(self.win.width() - t.width() - 24, y)
            y += t.height() + 10


class ErrorDialog(QDialog):
    def __init__(self, parent, L, summary, details):
        super().__init__(parent)
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(640, 420)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(12)
        head = QHBoxLayout()
        ic = QLabel("⚠")
        ic.setStyleSheet(f"font-size:28px; color:{PAL['danger']};")
        head.addWidget(ic)
        col = QVBoxLayout()
        t = QLabel(L("err_title"))
        t.setObjectName("cardTitle")
        s = QLabel(L("err_sub"))
        s.setObjectName("cardSub")
        s.setWordWrap(True)
        col.addWidget(t)
        col.addWidget(s)
        head.addLayout(col, 1)
        lay.addLayout(head)
        sm = QLabel(summary)
        sm.setWordWrap(True)
        sm.setStyleSheet(f"font-weight:600; color:{PAL['danger']};")
        lay.addWidget(sm)
        box = QTextEdit()
        box.setObjectName("console")
        box.setReadOnly(True)
        box.setPlainText(details)
        box.setFont(QFont("Consolas", 10))
        lay.addWidget(box, 1)
        row = QHBoxLayout()
        b_dir = QPushButton(L("err_logdir"))
        b_dir.clicked.connect(lambda: open_path(app_data_dir()))
        b_copy = QPushButton(L("err_copy"))
        b_copy.clicked.connect(lambda: QApplication.clipboard().setText(details))
        b_close = QPushButton(L("err_close"))
        b_close.setObjectName("primary")
        b_close.clicked.connect(self.accept)
        row.addWidget(b_dir)
        row.addStretch()
        row.addWidget(b_copy)
        row.addWidget(b_close)
        lay.addLayout(row)


def open_path(p):
    try:
        if os.name == "nt":
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{p}"')
        else:
            os.system(f'xdg-open "{p}"')
    except Exception:
        pass


# =============================================================================
# Arka plan işçisi
# =============================================================================
class StopRequested(Exception):
    pass


STEP_STATUS = {
    1: ("Görseller indiriliyor…", "Downloading images…"),
    2: ("YOLO ile etiketleniyor…", "Labeling with YOLO…"),
    3: ("Hatalı etiketler temizleniyor…", "Cleaning bad labels…"),
    4: ("Dosyalar ve plakalar yeniden adlandırılıyor…", "Renaming files and plates…"),
    5: ("Train / Val / Test ayrılıyor…", "Splitting Train / Val / Test…"),
}
CF_TITLES = ("just a moment", "bir dakika", "attention required", "checking your browser", "einen moment")


class PipelineWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(str)
    wait_input_signal = Signal(str)
    progress_signal = Signal(int, int)
    step_state_signal = Signal(int, str)
    finished_signal = Signal(str, str)  # status: done | cancelled | error

    def __init__(self, steps, config, session: SessionStore):
        super().__init__()
        self.steps = steps
        self.cfg = config
        self.session = session
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.is_cancelled = False
        self._proceed = False
        self._cur = None
        self.resume = bool(config.get("resume"))

    # --- yardımcılar ---
    def t(self, tr, en):
        return tr if self.cfg.get("lang", "tr") == "tr" else en

    def log(self, text):
        self.log_signal.emit(str(text))

    def progress(self, cur, total):
        self.progress_signal.emit(int(cur), int(total))

    def check_cancel(self):
        if self.is_cancelled:
            raise StopRequested()

    def cancel(self):
        self.is_cancelled = True
        self.mutex.lock()
        self.condition.wakeAll()
        self.mutex.unlock()

    def wait_for_user(self, prompt_msg):
        self.log(f"\n[{self.t('BEKLENİYOR', 'WAITING')}] {prompt_msg}")
        self._proceed = False
        self.wait_input_signal.emit(prompt_msg)
        self.mutex.lock()
        while not self._proceed and not self.is_cancelled:
            self.condition.wait(self.mutex, 300)
        self.mutex.unlock()
        self.check_cancel()

    def resume_from_input(self):
        self.mutex.lock()
        self._proceed = True
        self.condition.wakeAll()
        self.mutex.unlock()

    # --- ana döngü ---
    def run(self):
        try:
            for step in self.steps:
                self.check_cancel()
                self._cur = step
                self.session.set_current(step)
                self.step_state_signal.emit(step, "running")
                self.progress(0, 0)
                tr, en = STEP_STATUS[step]
                self.status_signal.emit(self.t(f"Adım {step}/5: {tr}", f"Step {step}/5: {en}"))
                getattr(self, f"run_step_{step}")()
                self.session.mark_completed(step)
                self.step_state_signal.emit(step, "done")
            self.session.finish("done")
            self.finished_signal.emit("done", "")
        except StopRequested:
            self.session.save()
            self.session.finish("interrupted")
            if self._cur:
                self.step_state_signal.emit(self._cur, "paused")
            self.log(self.t("\n[DURDURULDU] İlerleme kaydedildi.", "\n[STOPPED] Progress saved."))
            self.finished_signal.emit("cancelled", "")
        except Exception:
            tb = traceback.format_exc()
            write_crash_log(tb)
            self.log(f"\n[{self.t('KRİTİK HATA', 'CRITICAL ERROR')}]\n{tb}")
            self.session.finish("error")
            if self._cur:
                self.step_state_signal.emit(self._cur, "error")
            self.finished_signal.emit("error", tb)

    # ------------------------------------------------------------------
    # 1. ADIM: GÖRSEL İNDİRME
    # ------------------------------------------------------------------
    def run_step_1(self):
        global Camoufox, PlaywrightError
        if Camoufox is None:
            try:
                from camoufox.sync_api import Camoufox as _Camoufox
                from playwright.sync_api import Error as _PlaywrightError
                Camoufox = _Camoufox
                PlaywrightError = _PlaywrightError
            except ImportError as e:
                raise ImportError(self.t(f"Camoufox veya Playwright yüklü değil! ({e})",
                                         f"Camoufox or Playwright is not installed! ({e})"))
        self.log("=" * 60)
        self.log(self.t(">>> 1. ADIM BAŞLADI: Görsel İndirme İşlemi", ">>> STEP 1 STARTED: Image Download"))
        self.log("=" * 60)

        SCRIPT_KLASORU = self.cfg["base_dir"]
        KAYIT_KLASORU = os.path.join(SCRIPT_KLASORU, self.cfg["raw_folder"])
        NOT_DOSYASI = os.path.join(SCRIPT_KLASORU, self.cfg["s1_notes_file"])
        DURUM_DOSYASI = os.path.join(SCRIPT_KLASORU, self.cfg["s1_state_file"])
        BASLANGIC_SAYFASI = self.cfg["s1_start_page"]
        KAC_SAYFA_TARANACAK = self.cfg["s1_page_count"]
        BEKLEME_SURESI = self.cfg["s1_wait_time"]
        SAYFA_YUKLEME_ZAMAN_ASIMI = 30000
        GORSEL_ZAMAN_ASIMI = 15000
        DENEME_SAYISI = 2
        ANA_URL = self.cfg["s1_url"]
        LINK_SECICI = self.cfg["s1_link_selector"]
        ANA_ICERIK_SECICI = ""
        GORSEL_SECICI = self.cfg["s1_img_selector"]

        os.makedirs(KAYIT_KLASORU, exist_ok=True)

        def durum_yukle():
            if os.path.exists(DURUM_DOSYASI):
                try:
                    with open(DURUM_DOSYASI, "r", encoding="utf-8") as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass
            return {}

        def durum_kaydet(durum):
            atomic_write_json(Path(DURUM_DOSYASI), durum)

        DURUM = durum_yukle()
        SIRA_NO = max(DURUM.values(), default=0)

        def plaka_ayikla(title):
            m = re.search(r'"([^"]+)"', title or "")
            if not m:
                return None
            return m.group(1).replace(" ", "").strip()

        def not_yaz(no, plaka):
            with open(NOT_DOSYASI, "a", encoding="utf-8") as f:
                f.write(f"img({no})--{plaka}--\n")

        def sayfa_yuklenmesini_bekle(page):
            try:
                page.wait_for_load_state("load", timeout=SAYFA_YUKLEME_ZAMAN_ASIMI)
            except PlaywrightError:
                pass

        def cf_bekle(page, max_s=45):
            """Cloudflare ara sayfası bitene kadar bekler (ilk sayfaların yanlış okunmasını önler)."""
            t0 = time.time()
            uyarildi = False
            while time.time() - t0 < max_s:
                self.check_cancel()
                try:
                    baslik = (page.title() or "").lower()
                except Exception:
                    baslik = ""
                if not any(k in baslik for k in CF_TITLES):
                    return True
                if not uyarildi:
                    self.log(self.t("  Cloudflare doğrulaması algılandı, bekleniyor…",
                                    "  Cloudflare challenge detected, waiting…"))
                    uyarildi = True
                time.sleep(1.0)
            return False

        def guvenli_hrefler(page, secici):
            for _ in range(DENEME_SAYISI):
                sayfa_yuklenmesini_bekle(page)
                try:
                    return page.eval_on_selector_all(secici, "els => els.map(e => e.href)")
                except PlaywrightError as e:
                    if "context was destroyed" in str(e) or "navigation" in str(e):
                        time.sleep(2)
                        continue
                    raise
            return []

        def linkleri_topla(page):
            secici = f"{ANA_ICERIK_SECICI} {LINK_SECICI}".strip()
            page.wait_for_selector(secici, state="attached", timeout=SAYFA_YUKLEME_ZAMAN_ASIMI)
            sirali = []
            for h in guvenli_hrefler(page, secici):
                h = h.split("#")[0]
                if "/nomer" in h and h not in sirali:
                    sirali.append(h)
            return sirali

        def adim_tespit(page):
            """Sayfalama adımını (start artışı) birkaç deneme ile güvenilir biçimde bulur."""
            for _ in range(3):
                self.check_cancel()
                hrefler = guvenli_hrefler(page, "a[href*='start=']")
                degerler = {int(m.group(1)) for h in hrefler if (m := re.search(r"start=(\d+)", h))}
                pozitif = sorted(v for v in degerler if v > 0)
                if pozitif:
                    return pozitif[0]
                time.sleep(1.5)
            return None

        def gorsel_indir(page, detay_url):
            nonlocal SIRA_NO
            foto_id = detay_url.rstrip("/").split("nomer")[-1]

            if foto_id in DURUM:
                return foto_id, "mevcut"

            page.goto(detay_url.replace("/nomer", "/foto"), wait_until="domcontentloaded")
            cf_bekle(page)
            plaka = plaka_ayikla(page.title())

            img = page.locator(GORSEL_SECICI).first
            img.wait_for(state="visible", timeout=GORSEL_ZAMAN_ASIMI)
            img_url = img.get_attribute("src")
            if not img_url:
                return foto_id, "src yok"

            if img_url.startswith("//"):
                img_url = "https:" + img_url
            elif not img_url.startswith("http"):
                img_url = "https://platesmania.com" + img_url

            response = page.context.request.get(img_url)
            if response.status != 200:
                return foto_id, f"HTTP {response.status}"

            SIRA_NO += 1
            dosya_yolu = os.path.join(KAYIT_KLASORU, f"img({SIRA_NO}).jpg")
            tmp_yolu = dosya_yolu + ".part"
            with open(tmp_yolu, "wb") as f:
                f.write(response.body())
            os.replace(tmp_yolu, dosya_yolu)

            if not plaka:
                plaka = "BULUNAMADI"
            not_yaz(SIRA_NO, plaka)

            DURUM[foto_id] = SIRA_NO
            durum_kaydet(DURUM)
            return foto_id, f"kaydedildi -> img({SIRA_NO}) {plaka}"

        prog = self.session.get_progress(1) if self.resume else {}
        ilk_sayfa = BASLANGIC_SAYFASI
        if prog.get("last_page"):
            ilk_sayfa = max(BASLANGIC_SAYFASI, int(prog["last_page"]))
            self.log(self.t(f"Kayıttan devam: Sayfa {ilk_sayfa}", f"Resuming from saved page {ilk_sayfa}"))
        son_sayfa = BASLANGIC_SAYFASI + KAC_SAYFA_TARANACAK

        with Camoufox(humanize=True) as browser:
            page = browser.new_page()
            page.goto(f"{ANA_URL}&start=0", wait_until="domcontentloaded")

            self.log(self.t("Gerekirse tarayıcıdan Cloudflare doğrulamasını geçin.",
                            "Solve the Cloudflare challenge in the browser if needed."))
            self.wait_for_user(self.t(
                "Galeri sayfası TAMAMEN açıldığında devam etmek için üstteki 'Devam Et / Onayla' butonuna basın…",
                "Once the gallery page is FULLY loaded, press the 'Continue / Confirm' button above…"))

            cf_bekle(page)
            sayfa_yuklenmesini_bekle(page)

            # İlk sayfadaki link sayısı: sayfalama adımı bulunamazsa güvenli yedek olarak kullanılır
            try:
                ilk_linkler = linkleri_topla(page)
            except Exception:
                ilk_linkler = []
            adim = adim_tespit(page)
            if adim is None or adim < 2:
                yedek = len(ilk_linkler) if len(ilk_linkler) >= 2 else (adim or 1)
                self.log(self.t(
                    f"Uyarı: sayfalama adımı güvenilir bulunamadı ({adim}), yedek değer kullanılıyor: {yedek}",
                    f"Warning: pagination step not reliable ({adim}), using fallback: {yedek}"))
                adim = yedek
            self.log(self.t(f"Tespit edilen sayfa adımı (start artışı): {adim}\n",
                            f"Detected page step (start increment): {adim}\n"))

            onceki_linkler = set()

            for sayfa_no in range(ilk_sayfa, son_sayfa):
                self.check_cancel()
                start = (sayfa_no - 1) * adim
                page.goto(f"{ANA_URL}&start={start}", wait_until="domcontentloaded")
                cf_bekle(page)

                try:
                    tum_linkler = linkleri_topla(page)
                except StopRequested:
                    raise
                except Exception as e:
                    self.log(self.t(f"--- SAYFA {sayfa_no}: link bulunamadı ({e}) ---",
                                    f"--- PAGE {sayfa_no}: no links found ({e}) ---"))
                    break

                yeni_linkler = [l for l in tum_linkler if l not in onceki_linkler]
                onceki_linkler = set(tum_linkler)

                if not yeni_linkler:
                    self.log(self.t(f"--- SAYFA {sayfa_no}: yeni araç yok, tarama durduruldu ---",
                                    f"--- PAGE {sayfa_no}: no new vehicles, scan stopped ---"))
                    break

                self.session.set_progress(1, last_page=sayfa_no, force=True)
                self.log(self.t(f"--- SAYFA {sayfa_no} (start={start}, {len(yeni_linkler)} araç) ---",
                                f"--- PAGE {sayfa_no} (start={start}, {len(yeni_linkler)} vehicles) ---"))

                for index, detay_url in enumerate(yeni_linkler, start=1):
                    self.check_cancel()
                    try:
                        foto_id, durum = gorsel_indir(page, detay_url)
                        self.log(f"  [{index}/{len(yeni_linkler)}] {foto_id}: {durum}")
                        if durum.startswith("kaydedildi"):
                            time.sleep(BEKLEME_SURESI)
                    except StopRequested:
                        raise
                    except Exception as e:
                        self.log(self.t(f"  [{index}/{len(yeni_linkler)}] Atlandı (Hata): {e}",
                                        f"  [{index}/{len(yeni_linkler)}] Skipped (Error): {e}"))

                self.progress(sayfa_no - BASLANGIC_SAYFASI + 1, KAC_SAYFA_TARANACAK)
                self.log("")

            self.log(self.t("1. Adım (İndirme) tamamlandı.", "Step 1 (Download) completed."))
            if len(self.steps) == 1:
                self.wait_for_user(self.t("Tarayıcıyı kapatmak için 'Devam Et / Onayla' butonuna basın…",
                                          "Press 'Continue / Confirm' to close the browser…"))

    # ------------------------------------------------------------------
    # 2. ADIM: YOLO İLE ETİKETLEME
    # ------------------------------------------------------------------
    def run_step_2(self):
        global torch, torchvision, cv2, YOLO
        if YOLO is None or cv2 is None:
            self.log(self.t("Yapay zeka motoru (PyTorch & YOLO) yükleniyor…",
                            "Loading AI engine (PyTorch & YOLO)…"))
            try:
                import torch as _torch
                import torchvision as _torchvision
                import cv2 as _cv2
                from ultralytics import YOLO as _YOLO
                torch = _torch
                torchvision = _torchvision
                cv2 = _cv2
                YOLO = _YOLO
            except ImportError as e:
                raise ImportError(self.t(f"ultralytics (YOLO), torch veya opencv-python (cv2) yüklü değil! ({e})",
                                         f"ultralytics (YOLO), torch or opencv-python (cv2) is not installed! ({e})"))
        self.log("\n" + "=" * 60)
        self.log(self.t(">>> 2. ADIM BAŞLADI: YOLO Etiketleme İşlemi", ">>> STEP 2 STARTED: YOLO Labeling"))
        self.log("=" * 60)

        BASE_DIR = Path(self.cfg["base_dir"])
        model_val = self.cfg["s2_model_path"]
        aday_1 = Path(model_val) if os.path.isabs(model_val) else BASE_DIR / model_val
        aday_2_str = resource_path(model_val)
        if aday_1.exists():
            MODEL_PATH = aday_1
        elif aday_2_str:
            MODEL_PATH = Path(aday_2_str)
        else:
            MODEL_PATH = aday_1

        IMAGE_DIR = BASE_DIR / self.cfg["raw_folder"]
        MANUAL_CLASS_ID = self.cfg["s2_class_id"]
        LABEL_CAR_DIR = BASE_DIR / self.cfg["drawn_folder"]
        ETIKETLER_DIR = BASE_DIR / self.cfg["labels_folder"]
        CONFIDENCE = self.cfg["s2_conf"]
        IMAGE_SIZE = self.cfg["s2_imgsz"]
        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model bulunamadı / not found: {MODEL_PATH}")
        if not IMAGE_DIR.is_dir():
            raise FileNotFoundError(f"Resim klasörü bulunamadı / not found: {IMAGE_DIR}")

        LABEL_CAR_DIR.mkdir(parents=True, exist_ok=True)
        ETIKETLER_DIR.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(MODEL_PATH))

        image_paths = sorted(
            p for p in IMAGE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            self.log(self.t("Klasörde işlenecek resim bulunamadı.", "No images to process."))
            return

        prog = self.session.get_progress(2)
        skip_existing = self.resume and bool(prog.get("started"))
        self.session.set_progress(2, force=True, started=True)

        total = len(image_paths)
        self.log(self.t(f"Toplam resim: {total}", f"Total images: {total}"))
        self.log(self.t(f"Model sınıfları: {model.names}", f"Model classes: {model.names}"))
        if skip_existing:
            self.log(self.t("Devam modu: etiketi tamamlanmış görseller atlanacak.",
                            "Resume mode: already labeled images will be skipped."))

        atlanan = 0
        for index, image_path in enumerate(image_paths, start=1):
            self.check_cancel()
            self.progress(index - 1, total)
            label_path = ETIKETLER_DIR / f"{image_path.stem}.txt"
            label_car_path = LABEL_CAR_DIR / image_path.name

            if skip_existing and label_path.exists() and label_car_path.exists():
                atlanan += 1
                continue

            self.log(f"[{index}/{total}] {self.t('İşleniyor', 'Processing')}: {image_path.name}")
            results = model.predict(source=str(image_path), conf=CONFIDENCE, imgsz=IMAGE_SIZE, verbose=False)
            result = results[0]

            annotated_image = result.plot()
            ok, buf = cv2.imencode(image_path.suffix.lower() or ".jpg", annotated_image)
            if not ok:
                raise RuntimeError(f"Görsel kaydedilemedi / could not encode: {label_car_path}")
            label_car_path.write_bytes(buf.tobytes())

            lines = []
            if result.boxes is not None:
                for box in result.boxes:
                    xc, yc, bw, bh = box.xywhn[0].tolist()
                    lines.append(f"{MANUAL_CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            # TXT en son ve atomik yazılır: varlığı "bu görsel tamamlandı" anlamına gelir
            atomic_write_text(label_path, "".join(lines))

            self.log(self.t(
                f"  Tespit: {len(lines)} | TXT: {label_path.name} | Görsel: {label_car_path.name}",
                f"  Detections: {len(lines)} | TXT: {label_path.name} | Image: {label_car_path.name}"))
            self.session.set_progress(2, last_index=index)

        self.progress(total, total)
        self.session.set_progress(2, force=True, last_index=total)
        if atlanan:
            self.log(self.t(f"Atlanan (zaten etiketli): {atlanan}", f"Skipped (already labeled): {atlanan}"))
        self.log(self.t("\nTüm görseller işlendi.", "\nAll images processed."))
        self.log(f"{self.t('Etiketlenmiş görseller', 'Annotated images')}: {LABEL_CAR_DIR}")
        self.log(f"{self.t('TXT etiketleri', 'TXT labels')}: {ETIKETLER_DIR}")

    # ------------------------------------------------------------------
    # 3. ADIM: HATALI ETİKET SİLİCİ
    # ------------------------------------------------------------------
    def run_step_3(self):
        self.log("\n" + "=" * 60)
        self.log(self.t(">>> 3. ADIM BAŞLADI: Hatalı Etiket Kontrolü ve Temizliği",
                        ">>> STEP 3 STARTED: Bad Label Check and Cleanup"))
        self.log("=" * 60)

        ANA_DIZIN = Path(self.cfg["base_dir"])
        ARABALAR_KLASORU = ANA_DIZIN / self.cfg["raw_folder"]
        ETIKETLER_KLASORU = ANA_DIZIN / self.cfg["labels_folder"]
        CIZIMLI_KLASOR = ANA_DIZIN / self.cfg["drawn_folder"]

        silinen_coklu = 0
        silinen_bos = 0

        if not ARABALAR_KLASORU.is_dir():
            raise FileNotFoundError(f"Resim klasörü bulunamadı / not found: {ARABALAR_KLASORU}")
        if not ETIKETLER_KLASORU.is_dir():
            raise FileNotFoundError(f"Etiket klasörü bulunamadı / not found: {ETIKETLER_KLASORU}")

        resimler = [p for p in ARABALAR_KLASORU.iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        toplam = len(resimler)

        for i, resim_yolu in enumerate(resimler, start=1):
            self.check_cancel()
            self.progress(i, toplam)
            resim_adi = resim_yolu.name
            txt_yolu = ETIKETLER_KLASORU / f"{resim_yolu.stem}.txt"
            cizimli_resim_yolu = CIZIMLI_KLASOR / resim_adi

            silinecek = False
            sebep = ""

            if txt_yolu.exists():
                with txt_yolu.open("r", encoding="utf-8") as f:
                    satirlar = f.readlines()
                gecerli = [s for s in satirlar if s.strip()]
                if len(gecerli) >= 2:
                    silinecek = True
                    sebep = self.t(f"{len(gecerli)} etiket var (Çoklu Tespit)",
                                   f"{len(gecerli)} labels (Multiple Detection)")
                    silinen_coklu += 1
                elif len(gecerli) == 0:
                    silinecek = True
                    sebep = self.t("Hiç etiket yok (Boş TXT)", "No labels (Empty TXT)")
                    silinen_bos += 1
            else:
                silinecek = True
                sebep = self.t("Txt dosyası bulunamadı (Boş)", "TXT file missing (Empty)")
                silinen_bos += 1

            if silinecek:
                for yol in (txt_yolu, resim_yolu, cizimli_resim_yolu):
                    if yol.exists():
                        yol.unlink()
                self.log(f"{self.t('Silindi', 'Deleted')}: {resim_adi} | {self.t('Sebep', 'Reason')}: {sebep}")

        self.log(self.t("\nTamamlandı!", "\nDone!"))
        self.log(f"{self.t('Çoklu etiketten silinen', 'Removed (multiple labels)')}: {silinen_coklu}")
        self.log(f"{self.t('Boş olduğu için silinen', 'Removed (empty)')}: {silinen_bos}")
        self.log(f"{self.t('Toplam silinen veri', 'Total removed')}: {silinen_coklu + silinen_bos}")

    # ------------------------------------------------------------------
    # 4. ADIM: EŞZAMANLI AD DEĞİŞTİRİCİ (günlüklü = yarıda kesilirse güvenle devam eder)
    # ------------------------------------------------------------------
    def run_step_4(self):
        self.log("\n" + "=" * 60)
        self.log(self.t(">>> 4. ADIM BAŞLADI: Yeniden Adlandırma & Plaka TXT Güncelleme",
                        ">>> STEP 4 STARTED: Renaming & Plate Notes Update"))
        self.log("=" * 60)

        ANA_DIZIN = self.cfg["base_dir"]
        klasor_gorsel_1 = os.path.join(ANA_DIZIN, self.cfg["raw_folder"])
        klasor_gorsel_2 = os.path.join(ANA_DIZIN, self.cfg["drawn_folder"])
        klasor_txt = os.path.join(ANA_DIZIN, self.cfg["labels_folder"])
        not_dosyasi = os.path.join(ANA_DIZIN, self.cfg["s1_notes_file"])
        journal_path = self.session.dir / "step4_journal.json"

        # Yarım kalmış bir günlük varsa önce onu tamamla (kalıcı güvenlik)
        journal = None
        if journal_path.exists():
            try:
                with open(journal_path, "r", encoding="utf-8") as f:
                    journal = json.load(f)
                self.log(self.t("Yarım kalmış yeniden adlandırma bulundu, tamamlanıyor…",
                                "Unfinished rename found, completing it…"))
            except (OSError, json.JSONDecodeError):
                journal = None

        if journal is None:
            klasor_listesi = [klasor_gorsel_1, klasor_gorsel_2, klasor_txt]
            haric = {x.strip() for x in self.cfg["s4_exclude"].split(",") if x.strip()}
            baslangic_no = self.cfg["s4_start_no"]

            if not os.path.exists(klasor_gorsel_1):
                raise FileNotFoundError(f"Orijinal görsel klasörü bulunamadı / not found: {klasor_gorsel_1}")

            gecerli_uzantilar = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
            benzersiz = set()
            for dosya in os.listdir(klasor_gorsel_1):
                tam = os.path.join(klasor_gorsel_1, dosya)
                if os.path.isfile(tam):
                    isim, uzanti = os.path.splitext(dosya)
                    if uzanti.lower() in gecerli_uzantilar and isim not in haric:
                        benzersiz.add(isim)

            def dogal_anahtar(metin):
                return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", metin)]

            sirali = sorted(benzersiz, key=dogal_anahtar)
            harita = {eski: f"img({i})" for i, eski in enumerate(sirali, start=baslangic_no)}
            self.log(self.t(f"Orijinal klasörde hayatta kalan {len(harita)} görsel baz alındı.\n",
                            f"{len(harita)} surviving images in the raw folder used as reference.\n"))

            ops = []
            for klasor in klasor_listesi:
                if not os.path.exists(klasor):
                    continue
                for dosya in os.listdir(klasor):
                    if os.path.isfile(os.path.join(klasor, dosya)):
                        isim, uzanti = os.path.splitext(dosya)
                        if isim in harita:
                            ops.append([klasor, dosya, f"__temp_{uuid.uuid4().hex}{uzanti}",
                                        harita[isim] + uzanti])

            # Plaka dosyasının yeni içeriği önceden hesaplanır (idempotent yazım için)
            notes_lines, silinen_plaka = None, 0
            if os.path.exists(not_dosyasi):
                with open(not_dosyasi, "r", encoding="utf-8") as f:
                    satirlar = f.readlines()
                plaka_sozlugu = {}
                for satir in satirlar:
                    s = satir.strip()
                    if not s:
                        continue
                    m = re.match(r"^(.+?)--(.*)--$", s)
                    if m:
                        plaka_sozlugu[m.group(1).strip()] = m.group(2).strip()
                notes_lines = []
                for eski in sirali:
                    yeni = harita[eski]
                    plaka = plaka_sozlugu.get(eski)
                    notes_lines.append(f"{yeni}--{plaka if plaka is not None else 'PLAKA_KAYDI_YOK'}--\n")
                silinen_plaka = sum(1 for k in plaka_sozlugu if k not in harita)

            journal = {"ops": ops, "notes_path": not_dosyasi, "notes_lines": notes_lines,
                       "removed_plates": silinen_plaka, "sample": list(harita.items())[:10]}
            atomic_write_json(journal_path, journal)

        ops = journal["ops"]
        toplam = max(1, len(ops) * 2)

        # 1. AŞAMA: geçici adlara
        for i, (klasor, orig, gecici, hedef) in enumerate(ops, start=1):
            self.check_cancel()
            src, tmp = os.path.join(klasor, orig), os.path.join(klasor, gecici)
            if os.path.exists(src) and not os.path.exists(tmp):
                os.rename(src, tmp)
            if i % 25 == 0 or i == len(ops):
                self.progress(i, toplam)

        # 2. AŞAMA: kalıcı adlara
        for i, (klasor, orig, gecici, hedef) in enumerate(ops, start=1):
            self.check_cancel()
            tmp, dst = os.path.join(klasor, gecici), os.path.join(klasor, hedef)
            if os.path.exists(tmp):
                os.replace(tmp, dst)
            if i % 25 == 0 or i == len(ops):
                self.progress(len(ops) + i, toplam)

        self.log(self.t("Dosya adlandırma tamamlandı! Örnek eşleşmeler:", "Renaming completed! Sample mappings:"))
        for eski, yeni in journal.get("sample", []):
            self.log(f"  '{eski}' -> '{yeni}'")

        # 3. AŞAMA: plaka dosyası
        if journal.get("notes_lines") is not None:
            atomic_write_text(Path(journal["notes_path"]), "".join(journal["notes_lines"]))
            self.log(self.t(f"\nPlaka dosyası güncellendi: {os.path.basename(journal['notes_path'])}",
                            f"\nPlate notes updated: {os.path.basename(journal['notes_path'])}"))
            self.log(self.t(f"  • Korunan satır: {len(journal['notes_lines'])}",
                            f"  • Kept lines: {len(journal['notes_lines'])}"))
            self.log(self.t(f"  • 3. adımda silindiği için temizlenen satır: {journal['removed_plates']}",
                            f"  • Lines removed (deleted in step 3): {journal['removed_plates']}"))
        else:
            self.log(self.t("\n[BİLGİ] Plaka not dosyası bulunamadı, güncelleme atlandı.",
                            "\n[INFO] Plate notes file not found, update skipped."))

        try:
            journal_path.unlink()
        except OSError:
            pass
        self.progress(toplam, toplam)

    # ------------------------------------------------------------------
    # 5. ADIM: TRAIN / VAL / TEST
    # ------------------------------------------------------------------
    def run_step_5(self):
        self.log("\n" + "=" * 60)
        self.log(self.t(">>> 5. ADIM BAŞLADI: Train / Val / Test Veri Seti Ayırma",
                        ">>> STEP 5 STARTED: Train / Val / Test Split"))
        self.log("=" * 60)

        BASE_DIR = self.cfg["base_dir"]
        kaynaklar = []
        for img_f, txt_f in self.cfg["s5_sources"]:
            img_f, txt_f = img_f.strip(), txt_f.strip()
            if img_f and txt_f:
                kaynaklar.append({
                    "resim": img_f if os.path.isabs(img_f) else os.path.join(BASE_DIR, img_f),
                    "txt": txt_f if os.path.isabs(txt_f) else os.path.join(BASE_DIR, txt_f),
                })
        if not kaynaklar:
            raise ValueError(self.t("En az bir geçerli Resim ve TXT kaynak klasör çifti belirtmelisiniz!",
                                    "Specify at least one valid Image and TXT source folder pair!"))

        out_folder = self.cfg["s5_output_dir"].strip()
        base_dir = out_folder if os.path.isabs(out_folder) else os.path.join(BASE_DIR, out_folder)
        train_orani = self.cfg["s5_train_ratio"]
        val_orani = self.cfg["s5_val_ratio"]

        for bolum in ("train", "val", "test"):
            os.makedirs(os.path.join(base_dir, "images", bolum), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "labels", bolum), exist_ok=True)

        tum_veriler = []
        for kaynak in kaynaklar:
            for klasor in (kaynak["resim"], kaynak["txt"]):
                if not os.path.isdir(klasor):
                    raise FileNotFoundError(f"Kaynak klasör bulunamadı / not found: {klasor}")
            for resim_adi in sorted(os.listdir(kaynak["resim"])):
                if resim_adi.lower().endswith((".png", ".jpg", ".jpeg")):
                    txt_yolu = os.path.join(kaynak["txt"], os.path.splitext(resim_adi)[0] + ".txt")
                    if os.path.exists(txt_yolu):
                        tum_veriler.append((os.path.join(kaynak["resim"], resim_adi), txt_yolu))

        prog = self.session.get_progress(5)
        seed = prog.get("seed") if (self.resume and prog.get("seed")) else random.randrange(1 << 30)
        self.session.set_progress(5, force=True, seed=seed)
        random.Random(seed).shuffle(tum_veriler)

        toplam = len(tum_veriler)
        train_sinir = int(toplam * train_orani)
        val_sinir = int(toplam * (train_orani + val_orani))
        bolumler = (("train", tum_veriler[:train_sinir]),
                    ("val", tum_veriler[train_sinir:val_sinir]),
                    ("test", tum_veriler[val_sinir:]))

        yapilan = 0
        for bolum, liste in bolumler:
            for resim_yolu, txt_yolu in liste:
                self.check_cancel()
                for src, alt in ((resim_yolu, "images"), (txt_yolu, "labels")):
                    dst = os.path.join(base_dir, alt, bolum, os.path.basename(src))
                    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
                        shutil.copy2(src, dst)
                yapilan += 1
                if yapilan % 20 == 0 or yapilan == toplam:
                    self.progress(yapilan, toplam)

        self.log(self.t(f"Bölme işlemi tamamlandı! Çıktı klasörü: {base_dir}",
                        f"Split completed! Output folder: {base_dir}"))
        self.log(f"{self.t('Toplam', 'Total')}: {toplam}")
        self.log(f"Train: {len(bolumler[0][1])}, Val: {len(bolumler[1][1])}, Test: {len(bolumler[2][1])}")


# =============================================================================
# Çökme koruması
# =============================================================================
class CrashBridge(QObject):
    crashed = Signal(str, str)


def install_exception_hooks(bridge: CrashBridge):
    def hook(et, ev, tb):
        if issubclass(et, KeyboardInterrupt):
            sys.__excepthook__(et, ev, tb)
            return
        text = "".join(traceback.format_exception(et, ev, tb))
        write_crash_log(text)
        bridge.crashed.emit("ui", text)

    def thread_hook(args):
        if args.exc_type is SystemExit:
            return
        text = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        write_crash_log(text)
        bridge.crashed.emit("thread", text)

    sys.excepthook = hook
    threading.excepthook = thread_hook


# =============================================================================
# Ana pencere
# =============================================================================
class MainWindow(QMainWindow):
    PAGE_COUNT = 7

    def __init__(self, bridge: CrashBridge):
        super().__init__()
        self.setMinimumSize(1120, 740)
        self.resize(1320, 840)
        self.worker: PipelineWorker | None = None
        self._i18n = []
        self._error_open = False
        self._saved_console_h = 230
        self._max_prev = None
        self._pulse_on = False
        self._started_at = 0.0
        self.console_collapsed = False

        self.data_dir = app_data_dir()
        self.prefs_path = self.data_dir / "settings.json"
        self.prefs = self._load_prefs()
        self.session = SessionStore(self.data_dir / "session.json")

        self.lang = self.prefs.get("lang", "tr")
        self.theme_name = self.prefs.get("theme", "dark")
        self.animations_enabled = bool(self.prefs.get("animations", True))
        self.confirm_delete = bool(self.prefs.get("confirm_delete", True))

        if getattr(sys, "frozen", False):
            self.default_base_dir = os.path.dirname(sys.executable)
        else:
            self.default_base_dir = os.path.dirname(os.path.abspath(__file__))

        icon_file = resource_path("logo.ico") or resource_path("autoplate.ico")
        self.app_icon = QIcon(icon_file) if icon_file else QIcon()
        if icon_file:
            self.setWindowIcon(self.app_icon)

        self.toasts = ToastHost(self)
        bridge.crashed.connect(self.on_crash)

        self._build_ui()
        self.apply_theme(self.theme_name, animate=False)
        self.retranslate()

        cfg = self.session.data.get("config") if self.session.pending_steps() else self.prefs.get("config")
        if cfg:
            self.apply_config(cfg)
        self.set_font_size(self.prefs.get("font", 10))

        geo = self.prefs.get("geometry")
        if geo:
            self.restoreGeometry(QByteArray.fromBase64(geo.encode()))

        self.go_to(0, animate=False)
        self._setup_shortcuts()
        self._setup_timers()
        QTimer.singleShot(150, self.check_pending_session)
        QTimer.singleShot(60, lambda: self.splitter.setSizes([max(300, self.height() - 330), 230]))
        self._ready = True

    # ------------------------------------------------------------------
    # i18n
    # ------------------------------------------------------------------
    def L(self, key, **kw):
        s = LANG[self.lang].get(key, key)
        return s.format(**kw) if kw else s

    def reg(self, obj, key, setter="setText"):
        self._i18n.append((obj, key, setter))
        getattr(obj, setter)(self.L(key))
        return obj

    def retranslate(self):
        for obj, key, setter in self._i18n:
            getattr(obj, setter)(self.L(key))
        sub = self.L("app_sub").strip()
        self.setWindowTitle(f"{APP_NAME} — {sub}" if sub else APP_NAME)
        for i in range(1, 6):
            self.chips[i].title.setText(self.L(f"chip{i}"))
        self._refresh_chip_texts()
        self.update_test_ratio_label()
        self.s5_src2_img.setPlaceholderText(self.L("ph_s5_img"))
        self.s5_src2_txt.setPlaceholderText(self.L("ph_s5_txt"))
        self.theme_combo.blockSignals(True)
        self.theme_combo.setItemText(0, self.L("theme_dark"))
        self.theme_combo.setItemText(1, self.L("theme_light"))
        self.theme_combo.blockSignals(False)
        for combo, a, b in ((self.anim_combo, "anim_on", "anim_off"), (self.confirm_combo, "yes", "no")):
            combo.blockSignals(True)
            combo.setItemText(0, self.L(a))
            combo.setItemText(1, self.L(b))
            combo.blockSignals(False)
        self.update_page_header(self.stack.currentIndex())
        if not (self.worker and self.worker.isRunning()):
            self.status_label.setText(self.L("status_ready"))
        self.refresh_autosave_label()

    # ------------------------------------------------------------------
    # Tercihler
    # ------------------------------------------------------------------
    def _load_prefs(self):
        try:
            with open(self.prefs_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def save_prefs(self):
        try:
            data = {
                "lang": self.lang, "theme": self.theme_name,
                "animations": self.animations_enabled, "confirm_delete": self.confirm_delete,
                "font": self.font_spin.value(), "config": self.get_config(),
                "geometry": bytes(self.saveGeometry().toBase64()).decode(),
            }
            atomic_write_json(self.prefs_path, data)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # UI iskeleti
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_sidebar())

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(26, 20, 26, 16)
        rl.setSpacing(14)
        outer.addWidget(right, 1)

        rl.addWidget(self._build_header())
        rl.addWidget(self._build_runbar())

        # İçerik + konsol (splitter)
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setChildrenCollapsible(True)
        self.splitter.setHandleWidth(10)
        self.stack = QStackedWidget()
        self.stack.setMinimumHeight(140)
        self.splitter.addWidget(self.stack)

        self.console = QTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(0)
        self.console.document().setMaximumBlockCount(6000)
        self.splitter.addWidget(self.console)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setCollapsible(0, False)
        self.splitter.splitterMoved.connect(self._on_splitter_moved)
        rl.addWidget(self.splitter, 1)
        rl.addWidget(self._build_console_bar())

        self._build_pages()

    def _build_sidebar(self):
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(250)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(14, 20, 14, 14)
        lay.setSpacing(6)

        brand = QHBoxLayout()
        brand.setSpacing(10)
        logo = QLabel()
        logo.setFixedSize(40, 40)
        logo.setAlignment(Qt.AlignCenter)
        pm = QPixmap(resource_path("logo.ico") or resource_path("autoplate.ico"))
        if pm.isNull():
            pm = self.app_icon.pixmap(256, 256)
        if not pm.isNull():
            logo.setPixmap(pm.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo.setText("AP")
            logo.setStyleSheet("background:#6366F1;color:white;border-radius:10px;font-weight:700;")
        brand.addWidget(logo)
        col = QVBoxLayout()
        col.setSpacing(0)
        name = QLabel(APP_NAME)
        name.setObjectName("brandName")
        sub = QLabel()
        sub.setObjectName("brandSub")
        self.reg(sub, "app_sub")
        col.addWidget(name)
        col.addWidget(sub)
        brand.addLayout(col, 1)
        lay.addLayout(brand)
        lay.addSpacing(16)

        self.nav = QWidget()
        nl = QVBoxLayout(self.nav)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.setSpacing(4)
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons: list[NavButton] = []

        def section(key):
            l = QLabel()
            l.setObjectName("navSection")
            l.setContentsMargins(8, 8, 0, 4)
            self.reg(l, key)
            nl.addWidget(l)

        def add_nav(i):
            b = NavButton()
            self.reg(b, f"nav{i}")
            b.clicked.connect(lambda _=False, idx=i: self.go_to(idx))
            self.nav_group.addButton(b)
            self.nav_buttons.append(b)
            nl.addWidget(b)

        add_nav(0)
        section("sec_flow")
        for i in range(1, 6):
            add_nav(i)
        section("sec_sys")
        add_nav(6)

        self.indicator = QFrame(self.nav)
        self.indicator.setObjectName("navIndicator")
        self.indicator.setGeometry(0, 0, 3, 20)
        self.indicator.raise_()
        lay.addWidget(self.nav)
        lay.addStretch(1)

        self.autosave_lbl = QLabel()
        self.autosave_lbl.setObjectName("footerLbl")
        ver = QLabel(f"v{APP_VERSION}")
        ver.setObjectName("footerLbl")
        lay.addWidget(self.autosave_lbl)
        lay.addWidget(ver)
        return side

    def _build_header(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        col = QVBoxLayout()
        col.setSpacing(2)
        self.page_title = QLabel()
        self.page_title.setObjectName("pageTitle")
        self.page_sub = QLabel()
        self.page_sub.setObjectName("pageSub")
        col.addWidget(self.page_title)
        col.addWidget(self.page_sub)
        self.header_text = QWidget()
        self.header_text.setLayout(col)
        h.addWidget(self.header_text, 1)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["🇹🇷  Türkçe", "🇬🇧  English"])
        self.lang_combo.setFixedWidth(140)
        self.lang_combo.setCurrentIndex(0 if self.lang == "tr" else 1)
        self.lang_combo.currentIndexChanged.connect(lambda i: self.set_language("tr" if i == 0 else "en"))
        h.addWidget(self.lang_combo)

        self.btn_theme = QPushButton("☀")
        self.btn_theme.setObjectName("iconBtn")
        self.btn_theme.setCursor(Qt.PointingHandCursor)
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.reg(self.btn_theme, "tip_theme", "setToolTip")
        h.addWidget(self.btn_theme)
        return w

    def _build_runbar(self):
        bar = QFrame()
        bar.setObjectName("runBar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 12, 14, 12)
        lay.setSpacing(14)

        col = QVBoxLayout()
        col.setSpacing(6)
        top = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setObjectName("statusLbl")
        self.elapsed_label = QLabel("00:00")
        self.elapsed_label.setObjectName("elapsedLbl")
        top.addWidget(self.status_label, 1)
        top.addWidget(self.elapsed_label)
        col.addLayout(top)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        col.addWidget(self.progress_bar)
        lay.addLayout(col, 1)

        self.btn_continue = GlowButton()
        self.btn_continue.setObjectName("warn")
        self.btn_continue.setEnabled(False)
        self.btn_continue.clicked.connect(self.on_continue_clicked)
        self.reg(self.btn_continue, "continue_btn")
        self.btn_stop = GlowButton()
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        self.reg(self.btn_stop, "stop_btn")
        lay.addWidget(self.btn_continue)
        lay.addWidget(self.btn_stop)

        self._prog_anim = QPropertyAnimation(self.progress_bar, b"value", self)
        self._prog_anim.setDuration(350)
        self._prog_anim.setEasingCurve(QEasingCurve.OutCubic)
        return bar

    def _build_console_bar(self):
        bar = QFrame()
        bar.setObjectName("consoleBar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(6)
        self.btn_console_toggle = QPushButton("▾")
        self.btn_console_toggle.setObjectName("ghost")
        self.btn_console_toggle.setFixedWidth(32)
        self.btn_console_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_console_toggle.clicked.connect(self.toggle_console)
        self.reg(self.btn_console_toggle, "tip_toggle", "setToolTip")
        lay.addWidget(self.btn_console_toggle)
        title = QLabel()
        title.setObjectName("consoleTitle")
        self.reg(title, "log_group")
        lay.addWidget(title)
        self.console_preview = QLabel("")
        self.console_preview.setObjectName("consolePreview")
        self.console_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.console_preview.setVisible(False)
        lay.addWidget(self.console_preview, 1)
        for key, cb in (("clear_log", self.clear_console), ("copy_log", self.copy_console)):
            b = QPushButton()
            b.setObjectName("ghost")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(cb)
            self.reg(b, key)
            lay.addWidget(b)
        self.btn_console_max = QPushButton("⤢")
        self.btn_console_max.setObjectName("ghost")
        self.btn_console_max.setFixedWidth(34)
        self.btn_console_max.setCursor(Qt.PointingHandCursor)
        self.btn_console_max.clicked.connect(self.maximize_console)
        self.reg(self.btn_console_max, "tip_max", "setToolTip")
        lay.addWidget(self.btn_console_max)
        return bar

    # ------------------------------------------------------------------
    # Sayfa iskeleti
    # ------------------------------------------------------------------
    def make_page(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 8, 12)
        lay.setSpacing(16)
        scroll.setWidget(body)
        self.stack.addWidget(scroll)
        return lay

    def make_card(self, title_key, sub_key=None):
        card = QFrame()
        card.setObjectName("card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(22, 20, 22, 22)
        cl.setSpacing(14)
        if title_key:
            t = QLabel()
            t.setObjectName("cardTitle")
            self.reg(t, title_key)
            cl.addWidget(t)
        if sub_key:
            s = QLabel()
            s.setObjectName("cardSub")
            s.setWordWrap(True)
            self.reg(s, sub_key)
            cl.addWidget(s)
        return card, cl

    def field(self, key, widget):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(6)
        lab = QLabel()
        lab.setObjectName("fieldLabel")
        self.reg(lab, key)
        l.addWidget(lab)
        l.addWidget(widget)
        return w

    def path_row(self, edit, cb):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(8)
        l.addWidget(edit, 1)
        b = QPushButton()
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(cb)
        self.reg(b, "browse_btn")
        l.addWidget(b)
        return w

    def grid(self, cols=3):
        g = QGridLayout()
        g.setHorizontalSpacing(16)
        g.setVerticalSpacing(14)
        for c in range(cols):
            g.setColumnStretch(c, 1)
        return g

    def run_button(self, key, steps):
        b = GlowButton()
        b.setObjectName("primary")
        b.setMinimumHeight(46)
        b.clicked.connect(lambda: self.start_pipeline(steps))
        self.reg(b, key)
        self.run_buttons.append(b)
        return b

    def _build_pages(self):
        self.run_buttons: list[QPushButton] = []
        self.chips: dict[int, StepChip] = {}
        self._build_page_overview()
        self._build_page_s1()
        self._build_page_s2()
        self._build_page_s3()
        self._build_page_s4()
        self._build_page_s5()
        self._build_page_settings()

    # --- Genel bakış ---
    def _build_page_overview(self):
        lay = self.make_page()

        # Devam banner'ı
        self.banner = QFrame()
        self.banner.setObjectName("banner")
        bl = QHBoxLayout(self.banner)
        bl.setContentsMargins(18, 14, 14, 14)
        bl.setSpacing(12)
        bcol = QVBoxLayout()
        bcol.setSpacing(2)
        self.banner_title = QLabel()
        self.banner_title.setObjectName("bannerTitle")
        self.reg(self.banner_title, "resume_title")
        self.banner_text = QLabel()
        self.banner_text.setWordWrap(True)
        bcol.addWidget(self.banner_title)
        bcol.addWidget(self.banner_text)
        bl.addLayout(bcol, 1)
        self.btn_resume = GlowButton()
        self.btn_resume.setObjectName("primary")
        self.btn_resume.setStyleSheet("padding:9px 16px;")
        self.btn_resume.clicked.connect(self.resume_session)
        self.reg(self.btn_resume, "resume_btn")
        self.btn_discard = QPushButton()
        self.btn_discard.setObjectName("ghost")
        self.btn_discard.clicked.connect(self.discard_session)
        self.reg(self.btn_discard, "discard_btn")
        bl.addWidget(self.btn_resume)
        bl.addWidget(self.btn_discard)
        self.banner.setVisible(False)
        self.banner.setMaximumHeight(0)
        lay.addWidget(self.banner)

        # Çalışma alanı
        card, cl = self.make_card("ws_title", "ws_sub")
        g = self.grid(3)
        self.base_dir_edit = QLineEdit(self.default_base_dir)
        g.addWidget(self.field("base_dir_lbl", self.path_row(self.base_dir_edit, self.browse_base_dir)), 0, 0, 1, 3)
        self.raw_folder_edit = QLineEdit("elektrikli_araclar")
        self.drawn_folder_edit = QLineEdit("label4")
        self.labels_folder_edit = QLineEdit("etiketler4")
        g.addWidget(self.field("raw_folder_lbl", self.raw_folder_edit), 1, 0)
        g.addWidget(self.field("drawn_folder_lbl", self.drawn_folder_edit), 1, 1)
        g.addWidget(self.field("labels_folder_lbl", self.labels_folder_edit), 1, 2)
        cl.addLayout(g)
        lay.addWidget(card)

        # Hat
        card, cl = self.make_card("pl_title", "pl_sub")
        row = QHBoxLayout()
        row.setSpacing(10)
        for i in range(1, 6):
            chip = StepChip(i)
            chip.clicked.connect(self.go_to)
            self.chips[i] = chip
            row.addWidget(chip, 1)
        cl.addLayout(row)
        self.btn_run_all = self.run_button("run_all_btn", [1, 2, 3, 4, 5])
        self.btn_run_all.setMinimumHeight(52)
        cl.addWidget(self.btn_run_all)
        lay.addWidget(card)
        lay.addStretch(1)

        self.raw_folder_edit.textChanged.connect(self.sync_step5_primary_source)
        self.labels_folder_edit.textChanged.connect(self.sync_step5_primary_source)

    # --- Adım 1 ---
    def _build_page_s1(self):
        lay = self.make_page()
        card, cl = self.make_card("t1")
        g = self.grid(3)
        self.s1_url = QLineEdit("https://platesmania.com/de/gallery.php?ctype=11&gal=de")
        g.addWidget(self.field("s1_url_lbl", self.s1_url), 0, 0, 1, 3)
        self.s1_start_page = QSpinBox()
        self.s1_start_page.setRange(1, 10000)
        self.s1_start_page.setValue(2)
        self.s1_page_count = QSpinBox()
        self.s1_page_count.setRange(1, 10000)
        self.s1_page_count.setValue(100)
        self.s1_wait_time = QDoubleSpinBox()
        self.s1_wait_time.setRange(0.0, 60.0)
        self.s1_wait_time.setValue(2.0)
        g.addWidget(self.field("s1_start_page", self.s1_start_page), 1, 0)
        g.addWidget(self.field("s1_page_count", self.s1_page_count), 1, 1)
        g.addWidget(self.field("s1_wait_time", self.s1_wait_time), 1, 2)
        self.s1_notes_file = QLineEdit("plakalar4.txt")
        self.s1_state_file = QLineEdit("kayit.json")
        g.addWidget(self.field("s1_notes_file", self.s1_notes_file), 2, 0, 1, 2)
        g.addWidget(self.field("s1_state_file", self.s1_state_file), 2, 2)
        self.s1_link_sel = QLineEdit("a[href*='/de/nomer']")
        self.s1_img_sel = QLineEdit("img.img-responsive.center-block")
        g.addWidget(self.field("s1_link_sel", self.s1_link_sel), 3, 0, 1, 1)
        g.addWidget(self.field("s1_img_sel", self.s1_img_sel), 3, 1, 1, 2)
        cl.addLayout(g)
        cl.addWidget(self.run_button("s1_btn", [1]))
        lay.addWidget(card)
        lay.addStretch(1)

    # --- Adım 2 ---
    def _build_page_s2(self):
        lay = self.make_page()
        card, cl = self.make_card("t2")
        g = self.grid(3)
        self.s2_model_path = QLineEdit("V2.pt")
        g.addWidget(self.field("s2_model_lbl", self.path_row_model(self.s2_model_path)), 0, 0, 1, 3)
        self.s2_class_id = QSpinBox()
        self.s2_class_id.setRange(0, 999)
        self.s2_class_id.setValue(1)
        self.s2_conf = QDoubleSpinBox()
        self.s2_conf.setRange(0.01, 1.0)
        self.s2_conf.setSingleStep(0.05)
        self.s2_conf.setValue(0.25)
        self.s2_imgsz = QSpinBox()
        self.s2_imgsz.setRange(128, 2048)
        self.s2_imgsz.setSingleStep(32)
        self.s2_imgsz.setValue(640)
        g.addWidget(self.field("s2_class_lbl", self.s2_class_id), 1, 0)
        g.addWidget(self.field("s2_conf_lbl", self.s2_conf), 1, 1)
        g.addWidget(self.field("s2_imgsz_lbl", self.s2_imgsz), 1, 2)
        cl.addLayout(g)
        cl.addWidget(self.run_button("s2_btn", [2]))
        lay.addWidget(card)
        lay.addStretch(1)

    def path_row_model(self, edit):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(8)
        l.addWidget(edit, 1)
        b = QPushButton()
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(self.browse_model)
        self.reg(b, "s2_model_btn")
        l.addWidget(b)
        return w

    # --- Adım 3 ---
    def _build_page_s3(self):
        lay = self.make_page()
        card, cl = self.make_card("t3")
        info = QLabel()
        info.setObjectName("hint")
        info.setWordWrap(True)
        self.reg(info, "s3_info")
        cl.addWidget(info)
        cl.addWidget(self.run_button("s3_btn", [3]))
        lay.addWidget(card)
        lay.addStretch(1)

    # --- Adım 4 ---
    def _build_page_s4(self):
        lay = self.make_page()
        card, cl = self.make_card("t4")
        g = self.grid(2)
        self.s4_start_no = QSpinBox()
        self.s4_start_no.setRange(1, 999999)
        self.s4_start_no.setValue(1)
        self.s4_exclude = QLineEdit("classes, notes")
        g.addWidget(self.field("s4_start_no", self.s4_start_no), 0, 0)
        g.addWidget(self.field("s4_exclude", self.s4_exclude), 0, 1)
        cl.addLayout(g)
        info = QLabel()
        info.setObjectName("hint")
        info.setWordWrap(True)
        self.reg(info, "s4_info")
        cl.addWidget(info)
        cl.addWidget(self.run_button("s4_btn", [4]))
        lay.addWidget(card)
        lay.addStretch(1)

    # --- Adım 5 ---
    def _build_page_s5(self):
        lay = self.make_page()
        card, cl = self.make_card("t5")
        g = self.grid(3)
        self.s5_output_dir = QLineEdit("czeck_dataset")
        g.addWidget(self.field("s5_out_dir", self.s5_output_dir), 0, 0, 1, 3)
        self.s5_train_ratio = QDoubleSpinBox()
        self.s5_train_ratio.setRange(0.05, 0.95)
        self.s5_train_ratio.setSingleStep(0.05)
        self.s5_train_ratio.setValue(0.70)
        self.s5_val_ratio = QDoubleSpinBox()
        self.s5_val_ratio.setRange(0.05, 0.95)
        self.s5_val_ratio.setSingleStep(0.05)
        self.s5_val_ratio.setValue(0.15)
        self.s5_test_label = QLabel()
        self.s5_test_label.setStyleSheet("font-weight:700;")
        g.addWidget(self.field("s5_train_ratio", self.s5_train_ratio), 1, 0)
        g.addWidget(self.field("s5_val_ratio", self.s5_val_ratio), 1, 1)
        tw = QWidget()
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(0, 22, 0, 0)
        tl.addWidget(self.s5_test_label)
        g.addWidget(tw, 1, 2)
        self.s5_train_ratio.valueChanged.connect(self.update_test_ratio_label)
        self.s5_val_ratio.valueChanged.connect(self.update_test_ratio_label)
        self.s5_src1_img = QLineEdit(self.raw_folder_edit.text())
        self.s5_src1_txt = QLineEdit(self.labels_folder_edit.text())
        self.s5_src2_img = QLineEdit("")
        self.s5_src2_txt = QLineEdit("")
        g.addWidget(self.field("s5_src1_img", self.s5_src1_img), 2, 0, 1, 1)
        g.addWidget(self.field("s5_src1_txt", self.s5_src1_txt), 2, 1, 1, 2)
        g.addWidget(self.field("s5_src2_img", self.s5_src2_img), 3, 0, 1, 1)
        g.addWidget(self.field("s5_src2_txt", self.s5_src2_txt), 3, 1, 1, 2)
        cl.addLayout(g)
        cl.addWidget(self.run_button("s5_btn", [5]))
        lay.addWidget(card)
        lay.addStretch(1)

    # --- Ayarlar ---
    def _build_page_settings(self):
        lay = self.make_page()

        card, cl = self.make_card("set_appearance")
        g = self.grid(3)
        self.set_lang_combo = QComboBox()
        self.set_lang_combo.addItems(["Türkçe", "English"])
        self.set_lang_combo.setCurrentIndex(0 if self.lang == "tr" else 1)
        self.set_lang_combo.currentIndexChanged.connect(lambda i: self.set_language("tr" if i == 0 else "en"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setCurrentIndex(0 if self.theme_name == "dark" else 1)
        self.theme_combo.currentIndexChanged.connect(lambda i: self.apply_theme("dark" if i == 0 else "light"))
        self.font_spin = QSpinBox()
        self.font_spin.setRange(8, 20)
        self.font_spin.setValue(10)
        self.font_spin.valueChanged.connect(self.set_font_size)
        g.addWidget(self.field("set_lang_lbl", self.set_lang_combo), 0, 0)
        g.addWidget(self.field("set_theme_lbl", self.theme_combo), 0, 1)
        g.addWidget(self.field("set_font_lbl", self.font_spin), 0, 2)
        cl.addLayout(g)
        lay.addWidget(card)

        card, cl = self.make_card("set_behavior")
        g = self.grid(3)
        self.anim_combo = QComboBox()
        self.anim_combo.addItems(["On", "Off"])
        self.anim_combo.setCurrentIndex(0 if self.animations_enabled else 1)
        self.anim_combo.currentIndexChanged.connect(lambda i: setattr(self, "animations_enabled", i == 0))
        self.confirm_combo = QComboBox()
        self.confirm_combo.addItems(["Yes", "No"])
        self.confirm_combo.setCurrentIndex(0 if self.confirm_delete else 1)
        self.confirm_combo.currentIndexChanged.connect(lambda i: setattr(self, "confirm_delete", i == 0))
        g.addWidget(self.field("set_anim_lbl", self.anim_combo), 0, 0)
        g.addWidget(self.field("set_confirm_lbl", self.confirm_combo), 0, 1)
        cl.addLayout(g)
        lay.addWidget(card)

        card, cl = self.make_card("set_data")
        row = QHBoxLayout()
        row.setSpacing(10)
        for key, cb in (("set_save_btn", self.save_settings_json), ("set_load_btn", self.load_settings_json),
                        ("set_open_data", lambda: open_path(self.data_dir)),
                        ("set_reset_session", self.reset_session)):
            b = QPushButton()
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(cb)
            self.reg(b, key)
            row.addWidget(b)
        cl.addLayout(row)
        lay.addWidget(card)
        lay.addStretch(1)

    # ------------------------------------------------------------------
    # Navigasyon & animasyonlar
    # ------------------------------------------------------------------
    def go_to(self, idx, animate=True):
        idx = int(idx)
        prev = self.stack.currentIndex()
        self.nav_buttons[idx].setChecked(True)
        self.stack.setCurrentIndex(idx)
        self.update_page_header(idx)
        self.move_indicator(animate and prev != idx)
        if animate and self.animations_enabled and prev != idx:
            fade_in(self.stack.currentWidget(), 280)
            fade_in(self.header_text, 320, 0.2)

    def update_page_header(self, idx):
        self.page_title.setText(self.L(f"t{idx}"))
        self.page_sub.setText(self.L(f"sub{idx}"))

    def move_indicator(self, animate=True):
        btn = self.nav_buttons[self.stack.currentIndex()]
        target = QRect(0, btn.y() + 10, 3, btn.height() - 20)
        if animate and self.animations_enabled and self.indicator.isVisible():
            a = QPropertyAnimation(self.indicator, b"geometry", self)
            a.setDuration(320)
            a.setStartValue(self.indicator.geometry())
            a.setEndValue(target)
            a.setEasingCurve(QEasingCurve.OutBack)
            self._ind_anim = a
            a.start()
        else:
            self.indicator.setGeometry(target)
        self.indicator.raise_()

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, lambda: self.move_indicator(False))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not getattr(self, "_ready", False):
            return
        self.toasts.reflow()
        self.move_indicator(False)
        self._update_preview()

    # --- Tema ---
    def apply_theme(self, name, animate=True):
        global PAL
        self.theme_name = name
        PAL = dict(THEMES[name])
        overlay = None
        if animate and self.animations_enabled and self.isVisible():
            overlay = QLabel(self)
            overlay.setPixmap(self.grab())
            overlay.setGeometry(self.rect())
            overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
            overlay.show()
            overlay.raise_()
        up, down = build_arrow_assets(name, PAL["muted"])
        QApplication.instance().setStyleSheet(build_qss(PAL, up, down))
        self.btn_theme.setText("☀" if name == "dark" else "🌙")
        self.theme_combo.blockSignals(True)
        self.theme_combo.setCurrentIndex(0 if name == "dark" else 1)
        self.theme_combo.blockSignals(False)
        for b in self.nav_buttons:
            b.update()
        if overlay is not None:
            eff = QGraphicsOpacityEffect(overlay)
            overlay.setGraphicsEffect(eff)
            a = QPropertyAnimation(eff, b"opacity", overlay)
            a.setDuration(420)
            a.setStartValue(1.0)
            a.setEndValue(0.0)
            a.setEasingCurve(QEasingCurve.InOutQuad)
            a.finished.connect(overlay.deleteLater)
            self._theme_anim = a
            a.start()

    def toggle_theme(self):
        self.apply_theme("light" if self.theme_name == "dark" else "dark")

    def set_language(self, code):
        if code == self.lang:
            return
        self.lang = code
        for c in (self.lang_combo, self.set_lang_combo):
            c.blockSignals(True)
            c.setCurrentIndex(0 if code == "tr" else 1)
            c.blockSignals(False)
        self.retranslate()

    def set_font_size(self, v):
        v = int(v)
        f = QFont()
        f.setFamilies(["Cascadia Mono", "Consolas", "Menlo", "monospace"])
        f.setPointSize(v)
        self.console.setFont(f)
        if hasattr(self, "font_spin") and self.font_spin.value() != v:
            self.font_spin.blockSignals(True)
            self.font_spin.setValue(v)
            self.font_spin.blockSignals(False)

    # ------------------------------------------------------------------
    # Konsol (splitter animasyonlu)
    # ------------------------------------------------------------------
    def _on_splitter_moved(self, *_):
        self._sync_console_state(self.splitter.sizes()[1])

    def _sync_console_state(self, h):
        self.console_collapsed = h < 24
        self.btn_console_toggle.setText("▸" if self.console_collapsed else "▾")
        self.console_preview.setVisible(self.console_collapsed)
        if not self.console_collapsed:
            self._saved_console_h = max(120, h)
        self._update_preview()

    def animate_console(self, target):
        sizes = self.splitter.sizes()
        total = sum(sizes)
        cur = sizes[1]
        target = max(0, min(int(target), total - 160))
        if hasattr(self, "_split_anim"):
            self._split_anim.stop()
        if not self.animations_enabled:
            self.splitter.setSizes([total - target, target])
            self._sync_console_state(target)
            return
        a = QVariantAnimation(self)
        a.setDuration(340)
        a.setStartValue(cur)
        a.setEndValue(target)
        a.setEasingCurve(QEasingCurve.InOutCubic)
        a.valueChanged.connect(lambda v: self.splitter.setSizes([max(0, total - int(v)), int(v)]))
        a.finished.connect(lambda: self._sync_console_state(target))
        self._split_anim = a
        a.start()

    def toggle_console(self):
        if self.console_collapsed:
            self.animate_console(self._saved_console_h)
        else:
            self._saved_console_h = max(120, self.splitter.sizes()[1])
            self.animate_console(0)

    def maximize_console(self):
        sizes = self.splitter.sizes()
        total = sum(sizes)
        if sizes[1] > total * 0.55 and self._max_prev is not None:
            self.animate_console(self._max_prev)
            self._max_prev = None
        else:
            self._max_prev = sizes[1] if sizes[1] > 24 else 230
            self.animate_console(int(total * 0.7))

    def clear_console(self):
        self.console.clear()
        self.console_preview.setText("")

    def copy_console(self):
        QApplication.clipboard().setText(self.console.toPlainText())
        self.toasts.show(self.L("copy_log") + " ✓", "success", 1800)

    def _update_preview(self):
        if not hasattr(self, "_last_log_line"):
            return
        fm = self.console_preview.fontMetrics()
        self.console_preview.setText(
            fm.elidedText(self._last_log_line, Qt.ElideRight, max(60, self.console_preview.width() - 8)))

    def append_log(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        for line in str(text).split("\n"):
            raw = line.rstrip()
            low = raw.lower()
            if not raw:
                self.console.append(" ")
                continue
            if raw.startswith("===="):
                color = "#4B5873"
            elif raw.startswith(">>>"):
                color = "#7DD3FC"
            elif any(k in low for k in ("hata", "error", "critical", "kritik", "traceback")):
                color = "#F87171"
            elif any(k in low for k in ("bekleniyor", "waiting", "uyarı", "warning", "durduruldu", "stopped")):
                color = "#FBBF24"
            elif any(k in low for k in ("silindi", "deleted")):
                color = "#FB923C"
            elif any(k in low for k in ("kaydedildi", "tamamlandı", "completed", "başarıyla", "done")):
                color = "#34D399"
            else:
                color = None
            body = _h.escape(raw).replace(" ", " ")
            style = f"color:{color};" if color else ""
            self.console.append(
                f'<span style="color:#4B5873;">{ts} </span><span style="{style}">{body}</span>')
            self._last_log_line = raw.strip()
        self.console.moveCursor(QTextCursor.End)
        if self.console_collapsed:
            self._update_preview()

    # ------------------------------------------------------------------
    # Zamanlayıcılar, kısayollar
    # ------------------------------------------------------------------
    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self.toggle_console)
        QShortcut(QKeySequence("Ctrl+T"), self, activated=self.toggle_theme)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=lambda: self.start_pipeline([1, 2, 3, 4, 5]))

    def _setup_timers(self):
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self._tick_elapsed)
        self.pulse_timer = QTimer(self)
        self.pulse_timer.timeout.connect(self._tick_pulse)
        self.pulse_timer.start(650)
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self._autosave)
        self.autosave_timer.start(20000)
        self.refresh_autosave_label()

    def _tick_elapsed(self):
        s = int(time.time() - self._started_at)
        self.elapsed_label.setText(f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}" if s >= 3600
                                   else f"{s // 60:02d}:{s % 60:02d}")

    def _tick_pulse(self):
        self._pulse_on = not self._pulse_on
        for i in range(1, 6):
            self.chips[i].set_pulse(self._pulse_on)
            self.nav_buttons[i].set_pulse(self._pulse_on)

    def _autosave(self):
        self.session.set_config(self.get_config())
        self.save_prefs()
        self.refresh_autosave_label()

    def refresh_autosave_label(self):
        self.autosave_lbl.setText(self.L("autosave", t=datetime.now().strftime("%H:%M:%S")))

    # ------------------------------------------------------------------
    # Yapılandırma
    # ------------------------------------------------------------------
    def get_config(self):
        return {
            "lang": self.lang,
            "base_dir": self.base_dir_edit.text().strip(),
            "raw_folder": self.raw_folder_edit.text().strip(),
            "drawn_folder": self.drawn_folder_edit.text().strip(),
            "labels_folder": self.labels_folder_edit.text().strip(),
            "s1_url": self.s1_url.text().strip(),
            "s1_start_page": self.s1_start_page.value(),
            "s1_page_count": self.s1_page_count.value(),
            "s1_wait_time": self.s1_wait_time.value(),
            "s1_notes_file": self.s1_notes_file.text().strip(),
            "s1_state_file": self.s1_state_file.text().strip(),
            "s1_link_selector": self.s1_link_sel.text().strip(),
            "s1_img_selector": self.s1_img_sel.text().strip(),
            "s2_model_path": self.s2_model_path.text().strip(),
            "s2_class_id": self.s2_class_id.value(),
            "s2_conf": self.s2_conf.value(),
            "s2_imgsz": self.s2_imgsz.value(),
            "s4_start_no": self.s4_start_no.value(),
            "s4_exclude": self.s4_exclude.text().strip(),
            "s5_output_dir": self.s5_output_dir.text().strip(),
            "s5_train_ratio": self.s5_train_ratio.value(),
            "s5_val_ratio": self.s5_val_ratio.value(),
            "s5_sources": [
                [self.s5_src1_img.text(), self.s5_src1_txt.text()],
                [self.s5_src2_img.text(), self.s5_src2_txt.text()],
            ],
        }

    def apply_config(self, c):
        def sv(widget, key, setter="setText"):
            if key in c:
                getattr(widget, setter)(c[key])
        sv(self.base_dir_edit, "base_dir")
        sv(self.raw_folder_edit, "raw_folder")
        sv(self.drawn_folder_edit, "drawn_folder")
        sv(self.labels_folder_edit, "labels_folder")
        sv(self.s1_url, "s1_url")
        sv(self.s1_start_page, "s1_start_page", "setValue")
        sv(self.s1_page_count, "s1_page_count", "setValue")
        sv(self.s1_wait_time, "s1_wait_time", "setValue")
        sv(self.s1_notes_file, "s1_notes_file")
        sv(self.s1_state_file, "s1_state_file")
        sv(self.s1_link_sel, "s1_link_selector")
        sv(self.s1_img_sel, "s1_img_selector")
        sv(self.s2_model_path, "s2_model_path")
        sv(self.s2_class_id, "s2_class_id", "setValue")
        sv(self.s2_conf, "s2_conf", "setValue")
        sv(self.s2_imgsz, "s2_imgsz", "setValue")
        sv(self.s4_start_no, "s4_start_no", "setValue")
        sv(self.s4_exclude, "s4_exclude")
        sv(self.s5_output_dir, "s5_output_dir")
        sv(self.s5_train_ratio, "s5_train_ratio", "setValue")
        sv(self.s5_val_ratio, "s5_val_ratio", "setValue")
        srcs = c.get("s5_sources")
        if srcs and len(srcs) >= 2:
            self.s5_src1_img.setText(srcs[0][0])
            self.s5_src1_txt.setText(srcs[0][1])
            self.s5_src2_img.setText(srcs[1][0])
            self.s5_src2_txt.setText(srcs[1][1])

    def save_settings_json(self):
        path, _ = QFileDialog.getSaveFileName(self, self.L("set_save_btn"),
                                              os.path.join(self.base_dir_edit.text(), "autoplate_config.json"),
                                              "JSON (*.json)")
        if path:
            atomic_write_json(Path(path), self.get_config())
            self.toasts.show(self.L("msg_saved"), "success")

    def load_settings_json(self):
        path, _ = QFileDialog.getOpenFileName(self, self.L("set_load_btn"), self.base_dir_edit.text(),
                                              "JSON (*.json)")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.apply_config(json.load(f))
            self.toasts.show(self.L("msg_loaded"), "success")

    def reset_session(self):
        self.session.clear()
        self.hide_banner()
        for i in range(1, 6):
            self.set_step_state(i, "idle")
        self.toasts.show(self.L("msg_reset"), "info")

    # --- küçük slotlar ---
    def sync_step5_primary_source(self):
        if hasattr(self, "s5_src1_img"):
            self.s5_src1_img.setText(self.raw_folder_edit.text())
        if hasattr(self, "s5_src1_txt"):
            self.s5_src1_txt.setText(self.labels_folder_edit.text())

    def update_test_ratio_label(self):
        rem = 1.0 - (self.s5_train_ratio.value() + self.s5_val_ratio.value())
        p = int(round(rem * 100))
        self.s5_test_label.setText(self.L("s5_test", p=p))
        color = PAL["accent"] if rem > 0.0001 else PAL["danger"]
        self.s5_test_label.setStyleSheet(f"font-weight:700; color:{color};")

    def browse_base_dir(self):
        folder = QFileDialog.getExistingDirectory(self, self.L("dlg_dir"), self.base_dir_edit.text())
        if folder:
            self.base_dir_edit.setText(folder)

    def browse_model(self):
        fp, _ = QFileDialog.getOpenFileName(self, self.L("dlg_model"), self.base_dir_edit.text(),
                                            "YOLO Model (*.pt)")
        if fp:
            self.s2_model_path.setText(fp)

    # ------------------------------------------------------------------
    # Adım durumları
    # ------------------------------------------------------------------
    def _state_text(self, state):
        return self.L({"idle": "st_idle", "running": "st_running", "done": "st_done",
                       "error": "st_error", "paused": "st_paused"}[state])

    def set_step_state(self, step, state):
        self._chip_states[step] = state
        self.chips[step].set_state(state, self._state_text(state))
        self.nav_buttons[step].set_state(state)

    @property
    def _chip_states(self):
        if not hasattr(self, "_cs"):
            self._cs = {i: "idle" for i in range(1, 6)}
        return self._cs

    def _refresh_chip_texts(self):
        for i in range(1, 6):
            self.chips[i].set_state(self._chip_states[i], self._state_text(self._chip_states[i]))

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    def show_banner(self, steps):
        names = ", ".join(f"{s}·{self.L(f'chip{s}')}" for s in steps)
        self.banner_text.setText(self.L("resume_text", steps=names))
        self.banner.setVisible(True)
        target = self.banner.sizeHint().height() + 4
        if self.animations_enabled:
            a = QPropertyAnimation(self.banner, b"maximumHeight", self)
            a.setDuration(380)
            a.setStartValue(0)
            a.setEndValue(target)
            a.setEasingCurve(QEasingCurve.OutCubic)
            a.finished.connect(lambda: self.banner.setMaximumHeight(16777215))
            self._banner_anim = a
            a.start()
        else:
            self.banner.setMaximumHeight(16777215)
        self.btn_resume.start_pulse()

    def hide_banner(self):
        self.btn_resume.stop_pulse()
        if not self.banner.isVisible():
            return
        if self.animations_enabled:
            a = QPropertyAnimation(self.banner, b"maximumHeight", self)
            a.setDuration(280)
            a.setStartValue(self.banner.height())
            a.setEndValue(0)
            a.setEasingCurve(QEasingCurve.InCubic)
            a.finished.connect(lambda: self.banner.setVisible(False))
            self._banner_anim = a
            a.start()
        else:
            self.banner.setMaximumHeight(0)
            self.banner.setVisible(False)

    def check_pending_session(self):
        pending = self.session.pending_steps()
        if not pending:
            return
        for s in self.session.completed_steps():
            self.set_step_state(s, "done")
        cur = self.session.data["pipeline"].get("current")
        if cur in pending:
            self.set_step_state(cur, "paused")
        self.show_banner(pending)
        self.toasts.show(self.L("pending_toast"), "warn", 6500)

    def resume_session(self):
        pending = self.session.pending_steps()
        if not pending:
            self.hide_banner()
            return
        self.go_to(0)
        self.start_pipeline(pending, resume=True)

    def discard_session(self):
        self.session.clear()
        for i in range(1, 6):
            self.set_step_state(i, "idle")
        self.hide_banner()

    # ------------------------------------------------------------------
    # Hat çalıştırma
    # ------------------------------------------------------------------
    def is_running(self):
        return bool(self.worker and self.worker.isRunning())

    def start_pipeline(self, steps, resume=False):
        if self.is_running():
            self.toasts.show(self.L("msg_busy"), "warn")
            return
        cfg = self.get_config()
        if not os.path.isdir(cfg["base_dir"]):
            self.toasts.show(self.L("msg_bad_dir", p=cfg["base_dir"]), "error")
            self.go_to(0)
            return
        if 5 in steps and cfg["s5_train_ratio"] + cfg["s5_val_ratio"] >= 1.0:
            self.toasts.show(self.L("msg_bad_ratio"), "error")
            self.go_to(5)
            return
        if 3 in steps and not resume and self.confirm_delete:
            r = QMessageBox.question(self, self.L("msg_confirm_title"), self.L("msg_confirm_del"),
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return

        cfg["resume"] = resume
        self.hide_banner()
        self.session.set_config(self.get_config())
        self.session.begin(steps if not resume else self.session.data["pipeline"]["steps"], resume)
        completed = set(self.session.completed_steps()) if resume else set()
        for i in range(1, 6):
            self.set_step_state(i, "done" if i in completed else "idle")

        self.worker = PipelineWorker(steps, cfg, self.session)
        self.worker.log_signal.connect(self.append_log)
        self.worker.status_signal.connect(self.status_label.setText)
        self.worker.wait_input_signal.connect(self.on_wait_user_input)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.step_state_signal.connect(self.set_step_state)
        self.worker.finished_signal.connect(self.on_pipeline_finished)

        for b in self.run_buttons:
            b.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._started_at = time.time()
        self.elapsed_label.setText("00:00")
        self.elapsed_timer.start(1000)
        self.set_progress_value(0, 0)
        self.worker.start()

    def on_progress(self, cur, total):
        self.set_progress_value(cur, total)

    def set_progress_value(self, cur, total):
        if total <= 0:
            self.progress_bar.setRange(0, 0)
            return
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 1000)
        target = int(cur / total * 1000)
        if self.animations_enabled:
            self._prog_anim.stop()
            self._prog_anim.setStartValue(self.progress_bar.value())
            self._prog_anim.setEndValue(target)
            self._prog_anim.start()
        else:
            self.progress_bar.setValue(target)

    def on_wait_user_input(self, msg):
        self.btn_continue.setEnabled(True)
        self.btn_continue.start_pulse()
        self.status_label.setText(self.L("status_wait"))
        self.toasts.show(msg, "warn", 7000)

    def on_continue_clicked(self):
        if self.worker:
            self.btn_continue.setEnabled(False)
            self.btn_continue.stop_pulse()
            self.worker.resume_from_input()

    def on_stop_clicked(self):
        if self.worker and self.worker.isRunning():
            self.btn_stop.setEnabled(False)
            self.worker.cancel()

    def on_pipeline_finished(self, status, details):
        self.elapsed_timer.stop()
        for b in self.run_buttons:
            b.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_continue.setEnabled(False)
        self.btn_continue.stop_pulse()
        self.status_label.setText(self.L("status_ready"))
        self.refresh_autosave_label()
        if status == "done":
            self.set_progress_value(1, 1)
            self.toasts.show(self.L("msg_done"), "success", 6000)
        elif status == "cancelled":
            self.toasts.show(self.L("msg_stopped"), "warn", 6000)
            self.check_pending_session_banner_only()
        else:
            self.toasts.show(self.L("msg_error"), "error", 7000)
            summary = details.strip().splitlines()[-1] if details.strip() else ""
            self.show_error_dialog(summary, details)
            self.check_pending_session_banner_only()

    def check_pending_session_banner_only(self):
        pending = self.session.pending_steps()
        if pending:
            self.show_banner(pending)

    # ------------------------------------------------------------------
    # Hata gösterimi (uygulama çökmez)
    # ------------------------------------------------------------------
    def show_error_dialog(self, summary, details):
        if self._error_open:
            return
        self._error_open = True
        try:
            dlg = ErrorDialog(self, self.L, summary, details)
            dlg.exec()
        finally:
            self._error_open = False

    def on_crash(self, origin, text):
        try:
            self.session.set_config(self.get_config())
            if self.session.data["pipeline"]["status"] == "running":
                self.session.finish("interrupted")
            self.save_prefs()
        except Exception:
            pass
        self.append_log(f"[ERROR/{origin}]\n{text}")
        self.toasts.show(self.L("err_title"), "error", 6000)
        summary = text.strip().splitlines()[-1] if text.strip() else ""
        QTimer.singleShot(0, lambda: self.show_error_dialog(summary, text))

    # ------------------------------------------------------------------
    def closeEvent(self, e):
        if self.is_running():
            r = QMessageBox.question(self, APP_NAME, self.L("msg_close_running"),
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                e.ignore()
                return
            self.worker.cancel()
            if not self.worker.wait(8000):
                self.worker.terminate()
                self.worker.wait(2000)
                self.session.finish("interrupted")
        try:
            self.session.set_config(self.get_config())
            self.save_prefs()
        except Exception:
            pass
        e.accept()


# =============================================================================
def main():
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("optima.autoplate.studio.v3")
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setStyle("Fusion")
    base_font = QFont()
    base_font.setFamilies(["Segoe UI Variable", "Segoe UI", "Inter", "SF Pro Text", "Helvetica Neue", "Arial"])
    base_font.setPointSize(10)
    app.setFont(base_font)
    icon_file = resource_path("logo.ico") or resource_path("autoplate.ico")
    if icon_file and os.path.exists(icon_file):
        app.setWindowIcon(QIcon(icon_file))

    bridge = CrashBridge()
    install_exception_hooks(bridge)

    window = MainWindow(bridge)
    if icon_file and os.path.exists(icon_file):
        window.setWindowIcon(QIcon(icon_file))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()