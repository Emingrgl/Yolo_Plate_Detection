import sys
import os
import re
import json
import time
import uuid
import random
import shutil
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QTabWidget, QTextEdit, QFileDialog, QGroupBox, QMessageBox, QFrame,
    QComboBox, QSplitter, QGraphicsOpacityEffect
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QMutex, QWaitCondition, 
    QPropertyAnimation, QEasingCurve, QVariantAnimation
)
from PySide6.QtGui import QFont, QTextCursor, QIcon

# 1. ve 2. Script için gerekli kütüphaneler (Çalışma anında kontrol edilir)
try:
    from camoufox.sync_api import Camoufox
    from playwright.sync_api import Error as PlaywrightError
except ImportError:
    Camoufox = None
    PlaywrightError = Exception

try:
    import cv2
    from ultralytics import YOLO
except ImportError:
    cv2 = None
    YOLO = None

# Dil Sözlüğü (Localization)
LANG = {
    'tr': {
        'window_title': "AutoPlate",
        'common_dir': "Ortak Çalışma Dizini ve Klasör İsimleri (1, 2, 3 ve 4. Adımlar İçin)",
        'base_dir_lbl': "Ana Çalışma Dizini:",
        'browse_btn': "Klasör Seç",
        'raw_folder_lbl': "Ham Görsel Klasörü:",
        'drawn_folder_lbl': "Çizimli Görsel Klasörü:",
        'labels_folder_lbl': "TXT Etiket Klasörü:",
        'run_all_btn': "🚀 TÜMÜNÜ SIRAYLA ÇALIŞTIR (1 ➔ 2 ➔ 3 ➔ 4 ➔ 5)",
        'continue_btn': "⏸️ Devam Et / Onayla (Cloudflare & Tarayıcı)",
        'status_ready': "Durum: Hazır",
        'status_wait': "Durum: Kullanıcı Onayı Bekleniyor...",
        'log_group': "Canlı İşlem Çıktısı (Konsol / Log)",
        'clear_log': "Logları Temizle",
        'toggle_log': "🖥️ Konsolu Gizle/Göster",
        
        # Tabs
        'tab1': "1. Görsel İndirici",
        'tab2': "2. YOLO Etiketleme",
        'tab3': "3. Hatalı Etiket Silici",
        'tab4': "4. Ad Değiştirici",
        'tab5': "5. Dataset Ayırıcı (Train/Val/Test)",
        'tab6': "⚙️ Ayarlar",
        
        # Tab 1
        's1_url_lbl': "Hedef Galeri URL:",
        's1_start_page': "Başlangıç Sayfası:",
        's1_page_count': "Taranacak Sayfa Sayısı:",
        's1_wait_time': "Bekleme Süresi (sn):",
        's1_notes_file': "Plaka Not Dosyası (.txt):",
        's1_state_file': "Durum Dosyası (.json):",
        's1_link_sel': "Link Seçici (CSS):",
        's1_img_sel': "Görsel Seçici (CSS):",
        's1_btn': "▶ Sadece 1. Scripti (Görsel İndirici) Çalıştır",
        
        # Tab 2
        's2_model_lbl': "YOLO Model Dosyası (.pt):",
        's2_model_btn': "Model Seç",
        's2_class_lbl': "Atanacak Class ID (Sınıf No):",
        's2_conf_lbl': "Confidence (Güven Eşiği):",
        's2_imgsz_lbl': "Görsel Boyutu (imgsz):",
        's2_btn': "▶ Sadece 2. Scripti (YOLO Etiketleme) Çalıştır",
        
        # Tab 3
        's3_info': (
            "Bu adım, üst panelde belirttiğiniz Ham Görsel, Çizimli Görsel ve TXT Etiket klasörlerini tarar.\n"
            "• Etiket dosyası boş olan (0 tespit) veya hiç bulunmayan görselleri,\n"
            "• İçinde 2 veya daha fazla etiket (Çoklu Tespit) bulunan görselleri\n"
            "her 3 klasörden de kalıcı olarak siler."
        ),
        's3_btn': "▶ Sadece 3. Scripti (Hatalı Etiket Silici) Çalıştır",
        
        # Tab 4
        's4_start_no': "Başlangıç Numarası (Örn: img(1) için 1):",
        's4_exclude': "Hariç Tutulacak Dosyalar (Virgülle):",
        's4_info': (
            "Not: Bu adım yalnızca Ham Görsel Klasöründe kalan (3. adımdan sağ çıkan) dosyaları baz alır.\n"
            "• Görselleri ve etiketleri çakışma olmadan img(N) formatında yeniden sıralar.\n"
            "• 1. adımda oluşturulan Plaka Not Dosyası (.txt) içindeki kayıtları da eşzamanlı günceller ve silinen görsellerin plakalarını temizler."
        ),
        's4_btn': "▶ Sadece 4. Scripti (Yeniden Adlandırıcı) Çalıştır",
        
        # Tab 5
        's5_out_dir': "Çıktı Dataset Klasör Adı:",
        's5_train_ratio': "Train Oranı (Örn: 0.70):",
        's5_val_ratio': "Val Oranı (Örn: 0.15):",
        's5_src1_img': "1. Kaynak Resim Klasörü:",
        's5_src1_txt': "1. Kaynak TXT Klasörü:",
        's5_src2_img': "2. Kaynak Resim (Opsiyonel):",
        's5_src2_txt': "2. Kaynak TXT (Opsiyonel):",
        's5_btn': "▶ Sadece 5. Scripti (Train/Val/Test Ayırıcı) Çalıştır",

        # Tab 6 (Settings)
        'set_lang_lbl': "Uygulama Dili / Language:",
        'set_font_lbl': "Konsol Yazı Boyutu (pt):",
        'set_anim_lbl': "Arayüz Animasyonları:",
        'set_save_btn': "💾 Ayarları JSON Olarak Kaydet",
        'set_load_btn': "📂 Ayarları JSON'dan Yükle",
    },
    'en': {
        'window_title': "YOLO Dataset Automation Studio",
        'common_dir': "Workspace Directory & Folder Names (For Steps 1, 2, 3, and 4)",
        'base_dir_lbl': "Base Directory:",
        'browse_btn': "Browse",
        'raw_folder_lbl': "Raw Image Folder:",
        'drawn_folder_lbl': "Annotated Image Folder:",
        'labels_folder_lbl': "TXT Labels Folder:",
        'run_all_btn': "🚀 RUN ALL SEQUENTIALLY (1 ➔ 2 ➔ 3 ➔ 4 ➔ 5)",
        'continue_btn': "⏸️ Continue / Confirm (Cloudflare & Browser)",
        'status_ready': "Status: Ready",
        'status_wait': "Status: Waiting for User Confirmation...",
        'log_group': "Live Execution Output (Console / Log)",
        'clear_log': "Clear Logs",
        'toggle_log': "🖥️ Toggle Console",
        
        # Tabs
        'tab1': "1. Image Downloader",
        'tab2': "2. YOLO Labeling",
        'tab3': "3. Bad Label Remover",
        'tab4': "4. Batch Renamer",
        'tab5': "5. Dataset Splitter (Train/Val/Test)",
        'tab6': "⚙️ Settings",
        
        # Tab 1
        's1_url_lbl': "Target Gallery URL:",
        's1_start_page': "Start Page:",
        's1_page_count': "Pages to Scan:",
        's1_wait_time': "Wait Time (sec):",
        's1_notes_file': "Plate Notes File (.txt):",
        's1_state_file': "State File (.json):",
        's1_link_sel': "Link Selector (CSS):",
        's1_img_sel': "Image Selector (CSS):",
        's1_btn': "▶ Run Step 1 Only (Image Downloader)",
        
        # Tab 2
        's2_model_lbl': "YOLO Model File (.pt):",
        's2_model_btn': "Select Model",
        's2_class_lbl': "Target Class ID:",
        's2_conf_lbl': "Confidence Threshold:",
        's2_imgsz_lbl': "Image Size (imgsz):",
        's2_btn': "▶ Run Step 2 Only (YOLO Labeling)",
        
        # Tab 3
        's3_info': (
            "This step scans the Raw, Annotated, and TXT Label folders specified above.\n"
            "• Permanently deletes images with empty or missing label files (0 detections),\n"
            "• Permanently deletes images with 2 or more labels (Multiple detections)\n"
            "from all 3 folders."
        ),
        's3_btn': "▶ Run Step 3 Only (Bad Label Remover)",
        
        # Tab 4
        's4_start_no': "Start Number (e.g., 1 for img(1)):",
        's4_exclude': "Exclude Files (Comma-separated):",
        's4_info': (
            "Note: This step uses the surviving files in the Raw Image Folder as the reference.\n"
            "• Safely renames images and labels to img(N) using temporary UUIDs.\n"
            "• Simultaneously updates the Plate Notes File (.txt) from Step 1 and removes entries of deleted images."
        ),
        's4_btn': "▶ Run Step 4 Only (Batch Renamer)",
        
        # Tab 5
        's5_out_dir': "Output Dataset Folder Name:",
        's5_train_ratio': "Train Ratio (e.g., 0.70):",
        's5_val_ratio': "Val Ratio (e.g., 0.15):",
        's5_src1_img': "Primary Image Folder:",
        's5_src1_txt': "Primary TXT Folder:",
        's5_src2_img': "Secondary Image (Optional):",
        's5_src2_txt': "Secondary TXT (Optional):",
        's5_btn': "▶ Run Step 5 Only (Train/Val/Test Splitter)",

        # Tab 6 (Settings)
        'set_lang_lbl': "Application Language:",
        'set_font_lbl': "Console Font Size (pt):",
        'set_anim_lbl': "UI Animations:",
        'set_save_btn': "💾 Save Config to JSON",
        'set_load_btn': "📂 Load Config from JSON",
    }
}


class PipelineWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(str)
    wait_input_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, steps, config):
        super().__init__()
        self.steps = steps
        self.cfg = config
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.is_cancelled = False

    def log(self, text):
        self.log_signal.emit(str(text))

    def wait_for_user(self, prompt_msg):
        self.log(f"\n[BEKLENİYOR / WAITING] {prompt_msg}")
        self.wait_input_signal.emit(prompt_msg)
        self.mutex.lock()
        self.condition.wait(self.mutex)
        self.mutex.unlock()

    def resume_from_input(self):
        self.mutex.lock()
        self.condition.wakeAll()
        self.mutex.unlock()

    def run(self):
        try:
            for step in self.steps:
                if self.is_cancelled:
                    break
                if step == 1:
                    self.status_signal.emit("Adım 1/5: Görseller İndiriliyor... / Downloading...")
                    self.run_step_1()
                elif step == 2:
                    self.status_signal.emit("Adım 2/5: YOLO ile Etiketleniyor... / Labeling...")
                    self.run_step_2()
                elif step == 3:
                    self.status_signal.emit("Adım 3/5: Hatalı Etiketler Temizleniyor... / Cleaning...")
                    self.run_step_3()
                elif step == 4:
                    self.status_signal.emit("Adım 4/5: Dosyalar & Plakalar Yeniden Adlandırılıyor... / Renaming...")
                    self.run_step_4()
                elif step == 5:
                    self.status_signal.emit("Adım 5/5: Train / Val / Test Ayrılıyor... / Splitting...")
                    self.run_step_5()

            self.status_signal.emit("Hazır / Ready")
            self.finished_signal.emit(True, "Tüm seçili işlemler başarıyla tamamlandı!\nAll selected tasks completed successfully!")
        except Exception as e:
            self.log(f"\n[KRİTİK HATA / CRITICAL ERROR]: {str(e)}")
            self.status_signal.emit("Hata Oluştu! / Error!")
            self.finished_signal.emit(False, str(e))

    # 1. SCRIPT
    def run_step_1(self):
        if Camoufox is None:
            raise ImportError("Camoufox veya Playwright yüklü değil!")

        self.log("=" * 60)
        self.log(">>> 1. ADIM BAŞLADI: Görsel İndirme İşlemi")
        self.log("=" * 60)

        SCRIPT_KLASORU = self.cfg['base_dir']
        KAYIT_KLASORU = os.path.join(SCRIPT_KLASORU, self.cfg['raw_folder'])
        NOT_DOSYASI = os.path.join(SCRIPT_KLASORU, self.cfg['s1_notes_file'])
        DURUM_DOSYASI = os.path.join(SCRIPT_KLASORU, self.cfg['s1_state_file'])
        BASLANGIC_SAYFASI = self.cfg['s1_start_page']
        KAC_SAYFA_TARANACAK = self.cfg['s1_page_count']
        BEKLEME_SURESI = self.cfg['s1_wait_time']
        SAYFA_YUKLEME_ZAMAN_ASIMI = 30000
        GORSEL_ZAMAN_ASIMI = 15000
        DENEME_SAYISI = 2
        ANA_URL = self.cfg['s1_url']
        LINK_SECICI = self.cfg['s1_link_selector']
        ANA_ICERIK_SECICI = ""
        GORSEL_SECICI = self.cfg['s1_img_selector']

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
            with open(DURUM_DOSYASI, "w", encoding="utf-8") as f:
                json.dump(durum, f, ensure_ascii=False, indent=1)

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

        def adim_bul(page):
            hrefler = guvenli_hrefler(page, "a[href*='start=']")
            degerler = {int(m.group(1)) for h in hrefler if (m := re.search(r"start=(\d+)", h))}
            pozitif = sorted(v for v in degerler if v > 0)
            return pozitif[0] if pozitif else 1

        def gorsel_indir(page, detay_url):
            nonlocal SIRA_NO
            foto_id = detay_url.rstrip("/").split("nomer")[-1]

            if foto_id in DURUM:
                return foto_id, "mevcut"

            page.goto(detay_url.replace("/nomer", "/foto"), wait_until="domcontentloaded")
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
            with open(dosya_yolu, "wb") as f:
                f.write(response.body())

            if not plaka:
                plaka = "BULUNAMADI"
            not_yaz(SIRA_NO, plaka)

            DURUM[foto_id] = SIRA_NO
            durum_kaydet(DURUM)
            return foto_id, f"kaydedildi -> img({SIRA_NO}) {plaka}"

        with Camoufox(humanize=True) as browser:
            page = browser.new_page()
            page.goto(f"{ANA_URL}&start=0", wait_until="domcontentloaded")

            self.log("Gerekirse tarayıcıdan Cloudflare doğrulamasını geçin.")
            self.wait_for_user("Galeri sayfası TAMAMEN açıldığında devam etmek için üstteki 'Devam Et / Onayla' butonuna basın...")

            sayfa_yuklenmesini_bekle(page)
            adim = adim_bul(page)
            self.log(f"Tespit edilen sayfa adımı (start artışı): {adim}\n")

            onceki_linkler = set()

            for sayfa_no in range(BASLANGIC_SAYFASI, BASLANGIC_SAYFASI + KAC_SAYFA_TARANACAK):
                start = (sayfa_no - 1) * adim
                page.goto(f"{ANA_URL}&start={start}", wait_until="domcontentloaded")

                try:
                    tum_linkler = linkleri_topla(page)
                except Exception as e:
                    self.log(f"--- SAYFA {sayfa_no}: link bulunamadı ({e}) ---")
                    break

                yeni_linkler = [l for l in tum_linkler if l not in onceki_linkler]
                onceki_linkler = set(tum_linkler)

                if not yeni_linkler:
                    self.log(f"--- SAYFA {sayfa_no}: yeni araç yok, tarama durduruldu ---")
                    break

                self.log(f"--- SAYFA {sayfa_no} (start={start}, {len(yeni_linkler)} araç) ---")

                for index, detay_url in enumerate(yeni_linkler, start=1):
                    try:
                        foto_id, durum = gorsel_indir(page, detay_url)
                        self.log(f"  [{index}/{len(yeni_linkler)}] {foto_id}: {durum}")
                        if durum.startswith("kaydedildi"):
                            time.sleep(BEKLEME_SURESI)
                    except Exception as e:
                        self.log(f"  [{index}/{len(yeni_linkler)}] Atlandı (Hata): {e}")

                self.log("")

            self.log("1. Adım (İndirme) tamamlandı.")
            if len(self.steps) == 1:
                self.wait_for_user("Tarayıcıyı kapatmak için 'Devam Et / Onayla' butonuna basın...")

    # 2. SCRIPT: YOLO İLE LABEL OLUŞTURMA
    def run_step_2(self):
        if YOLO is None or cv2 is None:
            raise ImportError("ultralytics (YOLO) veya opencv-python (cv2) yüklü değil!")

        self.log("\n" + "=" * 60)
        self.log(">>> 2. ADIM BAŞLADI: YOLO Etiketleme İşlemi")
        self.log("=" * 60)

        BASE_DIR = Path(self.cfg['base_dir'])
        model_val = self.cfg['s2_model_path']
        MODEL_PATH = Path(model_val) if os.path.isabs(model_val) else BASE_DIR / model_val
        IMAGE_DIR = BASE_DIR / self.cfg['raw_folder']
        MANUAL_CLASS_ID = self.cfg['s2_class_id']
        LABEL_CAR_DIR = BASE_DIR / self.cfg['drawn_folder']
        ETIKETLER_DIR = BASE_DIR / self.cfg['labels_folder']
        CONFIDENCE = self.cfg['s2_conf']
        IMAGE_SIZE = self.cfg['s2_imgsz']

        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")
        if not IMAGE_DIR.is_dir():
            raise FileNotFoundError(f"Resim klasörü bulunamadı: {IMAGE_DIR}")

        LABEL_CAR_DIR.mkdir(parents=True, exist_ok=True)
        ETIKETLER_DIR.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(MODEL_PATH))

        image_paths = sorted(
            path for path in IMAGE_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_paths:
            self.log("Klasörde işlenecek resim bulunamadı.")
            return

        self.log(f"Toplam resim: {len(image_paths)}")
        self.log(f"Model sınıfları: {model.names}")

        for index, image_path in enumerate(image_paths, start=1):
            self.log(f"[{index}/{len(image_paths)}] İşleniyor: {image_path.name}")

            results = model.predict(
                source=str(image_path),
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                verbose=False
            )

            result = results[0]
            label_path = ETIKETLER_DIR / f"{image_path.stem}.txt"

            with label_path.open("w", encoding="utf-8") as file:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = MANUAL_CLASS_ID
                        x_center, y_center, box_width, box_height = box.xywhn[0].tolist()
                        file.write(
                            f"{class_id} "
                            f"{x_center:.6f} "
                            f"{y_center:.6f} "
                            f"{box_width:.6f} "
                            f"{box_height:.6f}\n"
                        )

            annotated_image = result.plot()
            label_car_path = LABEL_CAR_DIR / image_path.name

            success = cv2.imwrite(str(label_car_path), annotated_image)
            if not success:
                raise RuntimeError(f"Görsel kaydedilemedi: {label_car_path}")

            detection_count = len(result.boxes) if result.boxes is not None else 0
            self.log(f"  Tespit sayısı: {detection_count} | TXT: {label_path.name} | Görsel: {label_car_path.name}")

        self.log("\nTüm görseller işlendi.")
        self.log(f"Etiketlenmiş görseller: {LABEL_CAR_DIR}")
        self.log(f"TXT etiketleri: {ETIKETLER_DIR}")

    # 3. SCRIPT: HATALI LABEL KONTROL EDİCİ VE SİLİCİ
    def run_step_3(self):
        self.log("\n" + "=" * 60)
        self.log(">>> 3. ADIM BAŞLADI: Hatalı Etiket Kontrolü ve Temizliği")
        self.log("=" * 60)

        ANA_DIZIN = Path(self.cfg['base_dir'])
        ARABALAR_KLASORU = ANA_DIZIN / self.cfg['raw_folder']
        ETIKETLER_KLASORU = ANA_DIZIN / self.cfg['labels_folder']
        CIZIMLI_KLASOR = ANA_DIZIN / self.cfg['drawn_folder']

        silinen_coklu = 0
        silinen_bos = 0

        if not ARABALAR_KLASORU.is_dir():
            raise FileNotFoundError(f"Resim klasörü bulunamadı: {ARABALAR_KLASORU}")
        if not ETIKETLER_KLASORU.is_dir():
            raise FileNotFoundError(f"Etiket klasörü bulunamadı: {ETIKETLER_KLASORU}")

        for resim_yolu in list(ARABALAR_KLASORU.iterdir()):
            if resim_yolu.is_file() and resim_yolu.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                resim_adi = resim_yolu.name
                txt_yolu = ETIKETLER_KLASORU / f"{resim_yolu.stem}.txt"
                cizimli_resim_yolu = CIZIMLI_KLASOR / resim_adi

                silinecek = False
                sebep = ""

                if txt_yolu.exists():
                    with txt_yolu.open("r", encoding="utf-8") as f:
                        satirlar = f.readlines()

                    gecerli_satirlar = [satir for satir in satirlar if satir.strip()]

                    if len(gecerli_satirlar) >= 2:
                        silinecek = True
                        sebep = f"{len(gecerli_satirlar)} etiket var (Çoklu Tespit)"
                        silinen_coklu += 1
                    elif len(gecerli_satirlar) == 0:
                        silinecek = True
                        sebep = "Hiç etiket yok (Boş TXT)"
                        silinen_bos += 1
                else:
                    silinecek = True
                    sebep = "Txt dosyası bulunamadı (Boş)"
                    silinen_bos += 1

                if silinecek:
                    if txt_yolu.exists():
                        txt_yolu.unlink()
                    if resim_yolu.exists():
                        resim_yolu.unlink()
                    if cizimli_resim_yolu.exists():
                        cizimli_resim_yolu.unlink()

                    self.log(f"Silindi: {resim_adi} | Sebep: {sebep}")

        self.log("\nBitti La!")
        self.log(f"Çoklu etiketten giden: {silinen_coklu}")
        self.log(f"Boş yapılmış: {silinen_bos}")
        self.log(f"Toplam silinen veri: {silinen_coklu + silinen_bos}")

    # 4. SCRIPT: EŞZAMANLI AD DEĞİŞTİRİCİ + PLAKALAR.TXT SENKRONİZASYONU
    def run_step_4(self):
        self.log("\n" + "=" * 60)
        self.log(">>> 4. ADIM BAŞLADI: Eşzamanlı Yeniden Adlandırma & Plaka TXT Güncelleme")
        self.log("=" * 60)

        ANA_DIZIN = self.cfg['base_dir']
        klasor_gorsel_1 = os.path.join(ANA_DIZIN, self.cfg['raw_folder'])
        klasor_gorsel_2 = os.path.join(ANA_DIZIN, self.cfg['drawn_folder'])
        klasor_txt = os.path.join(ANA_DIZIN, self.cfg['labels_folder'])
        not_dosyasi = os.path.join(ANA_DIZIN, self.cfg['s1_notes_file'])

        klasor_listesi = [klasor_gorsel_1, klasor_gorsel_2, klasor_txt]
        haric_tutulacaklar = {x.strip() for x in self.cfg['s4_exclude'].split(",") if x.strip()}
        baslangic_no = self.cfg['s4_start_no']

        if not os.path.exists(klasor_gorsel_1):
            raise FileNotFoundError(f"[HATA] Orijinal görsel klasörü bulunamadı: {klasor_gorsel_1}")

        # KRİTİK: Sadece orijinal görsel klasöründe (3. adımdan sağ çıkan) dosyaları baz al!
        gecerli_uzantilar = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        benzersiz_isimler = set()

        for dosya in os.listdir(klasor_gorsel_1):
            tam_yol = os.path.join(klasor_gorsel_1, dosya)
            if os.path.isfile(tam_yol):
                isim, uzanti = os.path.splitext(dosya)
                if uzanti.lower() in gecerli_uzantilar and isim not in haric_tutulacaklar:
                    benzersiz_isimler.add(isim)

        # Doğal Sıralama (Natural Sort): img(2) -> img(10)'dan önce gelsin diye
        def dogal_anahtar(metin):
            return [int(parca) if parca.isdigit() else parca.lower() for parca in re.split(r'(\d+)', metin)]

        sirali_isimler = sorted(list(benzersiz_isimler), key=dogal_anahtar)
        isim_haritasi = {eski_isim: f"img({i})" for i, eski_isim in enumerate(sirali_isimler, start=baslangic_no)}

        self.log(f"Orijinal klasörde hayatta kalan {len(isim_haritasi)} adet görsel baz alındı. İşlem başlıyor...\n")

        gecici_dosyalar = []

        # 1. AŞAMA: Görsel ve Etiket Dosyalarını Geçici (UUID) Adlara Çevir
        for klasor in klasor_listesi:
            if not os.path.exists(klasor):
                continue
            for dosya in os.listdir(klasor):
                tam_yol = os.path.join(klasor, dosya)
                if os.path.isfile(tam_yol):
                    isim, uzanti = os.path.splitext(dosya)

                    if isim in isim_haritasi:
                        yeni_isim = isim_haritasi[isim] + uzanti
                        gecici_isim = f"__temp_{uuid.uuid4().hex}{uzanti}"
                        gecici_yol = os.path.join(klasor, gecici_isim)
                        hedef_yol = os.path.join(klasor, yeni_isim)

                        os.rename(tam_yol, gecici_yol)
                        gecici_dosyalar.append((gecici_yol, hedef_yol, dosya, yeni_isim))

        # 2. AŞAMA: Geçici Dosyaları Kalıcı Yeni Adlarına Çevir
        for gecici_yol, hedef_yol, eski_ad, yeni_ad in gecici_dosyalar:
            os.rename(gecici_yol, hedef_yol)

        self.log("Dosya adlandırma tamamlandı! Örnek eşleşmeler:")
        for eski, yeni in list(isim_haritasi.items())[:10]:
            self.log(f"  '{eski}' -> '{yeni}'")

        # 3. AŞAMA: PLAKALAR.TXT DOSYASINI SENKRONİZE ET VE SİLİNENLERİ TEMİZLE
        if os.path.exists(not_dosyasi):
            self.log(f"\nPlaka not dosyası güncelleniyor: {os.path.basename(not_dosyasi)}")
            with open(not_dosyasi, "r", encoding="utf-8") as f:
                satirlar = f.readlines()

            plaka_sozlugu = {}
            for satir in satirlar:
                satir_temiz = satir.strip()
                if not satir_temiz:
                    continue
                # Format: img(12)--34ABC123--
                m = re.match(r"^(.+?)--(.*)--$", satir_temiz)
                if m:
                    eski_gorsel_adi = m.group(1).strip()
                    plaka_metni = m.group(2).strip()
                    plaka_sozlugu[eski_gorsel_adi] = plaka_metni

            yeni_plaka_satirlari = []
            silinen_plaka_sayisi = 0

            # Orijinal klasördeki yeni sıralamaya göre plakaları diz
            for eski_isim in sirali_isimler:
                yeni_isim = isim_haritasi[eski_isim]
                if eski_isim in plaka_sozlugu:
                    plaka = plaka_sozlugu[eski_isim]
                    yeni_plaka_satirlari.append(f"{yeni_isim}--{plaka}--\n")
                else:
                    yeni_plaka_satirlari.append(f"{yeni_isim}--PLAKA_KAYDI_YOK--\n")

            # 3. adımda silinen görsellerin plaka kayıtlarını tespit et
            for eski_kayit in plaka_sozlugu:
                if eski_kayit not in isim_haritasi:
                    silinen_plaka_sayisi += 1

            with open(not_dosyasi, "w", encoding="utf-8") as f:
                f.writelines(yeni_plaka_satirlari)

            self.log(f"Plaka dosyası başarıyla güncellendi!")
            self.log(f"  • Güncellenip korunan plaka satırı: {len(yeni_plaka_satirlari)}")
            self.log(f"  • 3. adımda görseli silindiği için temizlenen plaka satırı: {silinen_plaka_sayisi}")
        else:
            self.log(f"\n[BİLGİ] Plaka not dosyası ({os.path.basename(not_dosyasi)}) bulunamadığı için TXT güncellemesi atlandı.")

    # 5. SCRIPT: TRAIN / VAL / TEST AYIRMA İŞLEMİ
    def run_step_5(self):
        self.log("\n" + "=" * 60)
        self.log(">>> 5. ADIM BAŞLADI: Train / Val / Test Veri Seti Ayırma")
        self.log("=" * 60)

        BASE_DIR = self.cfg['base_dir']
        kaynaklar = []

        for img_f, txt_f in self.cfg['s5_sources']:
            img_f = img_f.strip()
            txt_f = txt_f.strip()
            if img_f and txt_f:
                r_path = img_f if os.path.isabs(img_f) else os.path.join(BASE_DIR, img_f)
                t_path = txt_f if os.path.isabs(txt_f) else os.path.join(BASE_DIR, txt_f)
                kaynaklar.append({'resim': r_path, 'txt': t_path})

        if not kaynaklar:
            raise ValueError("En az bir geçerli Resim ve TXT kaynak klasör çifti belirtmelisiniz!")

        out_folder = self.cfg['s5_output_dir'].strip()
        base_dir = out_folder if os.path.isabs(out_folder) else os.path.join(BASE_DIR, out_folder)
        bolumler = ['train', 'val', 'test']

        train_orani = self.cfg['s5_train_ratio']
        val_orani = self.cfg['s5_val_ratio']

        for bolum in bolumler:
            os.makedirs(f"{base_dir}/images/{bolum}", exist_ok=True)
            os.makedirs(f"{base_dir}/labels/{bolum}", exist_ok=True)

        tum_veriler = []

        for kaynak in kaynaklar:
            r_klasor = kaynak['resim']
            e_klasor = kaynak['txt']

            for klasor in (r_klasor, e_klasor):
                if not os.path.isdir(klasor):
                    raise FileNotFoundError(f"Kaynak klasör bulunamadı: {klasor}")

            for resim_adi in os.listdir(r_klasor):
                if resim_adi.lower().endswith(('.png', '.jpg', '.jpeg')):
                    txt_adi = os.path.splitext(resim_adi)[0] + '.txt'
                    txt_yolu = os.path.join(e_klasor, txt_adi)
                    resim_yolu = os.path.join(r_klasor, resim_adi)

                    if os.path.exists(txt_yolu):
                        tum_veriler.append((resim_yolu, txt_yolu))

        random.shuffle(tum_veriler)
        toplam_veri = len(tum_veriler)

        train_sinir = int(toplam_veri * train_orani)
        val_sinir = int(toplam_veri * (train_orani + val_orani))

        train_data = tum_veriler[:train_sinir]
        val_data = tum_veriler[train_sinir:val_sinir]
        test_data = tum_veriler[val_sinir:]

        def dosyalari_kopyala(veri_listesi, bolum_adi):
            for resim_yolu, txt_yolu in veri_listesi:
                resim_isim = os.path.basename(resim_yolu)
                txt_isim = os.path.basename(txt_yolu)
                shutil.copy(resim_yolu, f"{base_dir}/images/{bolum_adi}/{resim_isim}")
                shutil.copy(txt_yolu, f"{base_dir}/labels/{bolum_adi}/{txt_isim}")

        dosyalari_kopyala(train_data, 'train')
        dosyalari_kopyala(val_data, 'val')
        dosyalari_kopyala(test_data, 'test')

        self.log(f"Bölme işlemi tamamlandı! Çıktı Klasörü: {base_dir}")
        self.log(f"Toplam: {toplam_veri} veri.")
        self.log(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


# GUI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_lang = 'tr'
        self.animations_enabled = True
        self.setWindowTitle(LANG[self.current_lang]['window_title'])
        self.setMinimumSize(1040, 780)
        self.worker = None

        # PyInstaller .exe desteği için çalışma dizini tespiti
        if getattr(sys, 'frozen', False):
            self.default_base_dir = os.path.dirname(sys.executable)
        else:
            self.default_base_dir = os.path.dirname(os.path.abspath(__file__))

        self.setup_ui()
        self.apply_stylesheet()

    def tr(self, key):
        return LANG[self.current_lang].get(key, key)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Üst Araç Çubuğu
        toolbar_layout = QHBoxLayout()
        
        self.app_title_lbl = QLabel("AutoPlate")
        self.app_title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #38bdf8;")
        toolbar_layout.addWidget(self.app_title_lbl)
        
        toolbar_layout.addStretch()

        self.btn_toggle_console = QPushButton(self.tr('toggle_log'))
        self.btn_toggle_console.setFixedWidth(170)
        self.btn_toggle_console.clicked.connect(self.toggle_console_animated)
        toolbar_layout.addWidget(self.btn_toggle_console)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["🇹🇷 Türkçe", "🇬🇧 English"])
        self.lang_combo.setFixedWidth(120)
        self.lang_combo.currentIndexChanged.connect(self.change_language)
        toolbar_layout.addWidget(self.lang_combo)

        main_layout.addLayout(toolbar_layout)

        # Ana Splitter (Kullanıcı boyutu fareyle uzatıp kısaltabilir)
        self.main_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(self.main_splitter, stretch=1)

        top_container = QWidget()
        top_container_layout = QVBoxLayout(top_container)
        top_container_layout.setContentsMargins(0, 0, 0, 4)

        # Ortak Klasörler Grubu
        self.top_group = QGroupBox(self.tr('common_dir'))
        top_layout = QGridLayout(self.top_group)

        self.lbl_base_dir = QLabel(self.tr('base_dir_lbl'))
        top_layout.addWidget(self.lbl_base_dir, 0, 0)
        self.base_dir_edit = QLineEdit(self.default_base_dir)
        top_layout.addWidget(self.base_dir_edit, 0, 1, 1, 4)
        self.btn_browse_base = QPushButton(self.tr('browse_btn'))
        self.btn_browse_base.clicked.connect(self.browse_base_dir)
        top_layout.addWidget(self.btn_browse_base, 0, 5)

        self.lbl_raw_folder = QLabel(self.tr('raw_folder_lbl'))
        top_layout.addWidget(self.lbl_raw_folder, 1, 0)
        self.raw_folder_edit = QLineEdit("elektrikli_araclar")
        self.raw_folder_edit.textChanged.connect(self.sync_step5_primary_source)
        top_layout.addWidget(self.raw_folder_edit, 1, 1)

        self.lbl_drawn_folder = QLabel(self.tr('drawn_folder_lbl'))
        top_layout.addWidget(self.lbl_drawn_folder, 1, 2)
        self.drawn_folder_edit = QLineEdit("label4")
        top_layout.addWidget(self.drawn_folder_edit, 1, 3)

        self.lbl_labels_folder = QLabel(self.tr('labels_folder_lbl'))
        top_layout.addWidget(self.lbl_labels_folder, 1, 4)
        self.labels_folder_edit = QLineEdit("etiketler4")
        self.labels_folder_edit.textChanged.connect(self.sync_step5_primary_source)
        top_layout.addWidget(self.labels_folder_edit, 1, 5)

        top_container_layout.addWidget(self.top_group)

        # Sekmeler ve Animasyon Bağlantısı
        self.tabs = QTabWidget()
        self.setup_tab_1()
        self.setup_tab_2()
        self.setup_tab_3()
        self.setup_tab_4()
        self.setup_tab_5()
        self.setup_tab_settings()
        self.tabs.currentChanged.connect(self.animate_tab_transition)
        top_container_layout.addWidget(self.tabs)

        # Aksiyon Çerçevesi
        action_frame = QFrame()
        action_frame.setObjectName("actionFrame")
        action_layout = QHBoxLayout(action_frame)

        self.btn_run_all = QPushButton(self.tr('run_all_btn'))
        self.btn_run_all.setObjectName("runAllBtn")
        self.btn_run_all.setFixedHeight(44)
        self.btn_run_all.clicked.connect(lambda: self.start_pipeline([1, 2, 3, 4, 5]))
        action_layout.addWidget(self.btn_run_all, stretch=2)

        self.btn_continue = QPushButton(self.tr('continue_btn'))
        self.btn_continue.setObjectName("continueBtn")
        self.btn_continue.setFixedHeight(44)
        self.btn_continue.setEnabled(False)
        self.btn_continue.clicked.connect(self.on_continue_clicked)
        action_layout.addWidget(self.btn_continue, stretch=1)

        self.status_label = QLabel(self.tr('status_ready'))
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_label.setStyleSheet("font-weight: bold; color: #38bdf8; padding-right: 8px;")
        action_layout.addWidget(self.status_label, stretch=1)

        top_container_layout.addWidget(action_frame)
        self.main_splitter.addWidget(top_container)

        # Alt Panel: Konsol / Log
        self.log_group = QGroupBox(self.tr('log_group'))
        log_layout = QVBoxLayout(self.log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 10))
        log_layout.addWidget(self.log_output)

        log_btn_layout = QHBoxLayout()
        log_btn_layout.addStretch()
        self.btn_clear_log = QPushButton(self.tr('clear_log'))
        self.btn_clear_log.setFixedWidth(140)
        self.btn_clear_log.clicked.connect(self.log_output.clear)
        log_btn_layout.addWidget(self.btn_clear_log)
        log_layout.addLayout(log_btn_layout)

        self.main_splitter.addWidget(self.log_group)
        self.main_splitter.setSizes([500, 260])

    def setup_tab_1(self):
        tab = QWidget()
        layout = QGridLayout(tab)

        self.s1_lbl_url = QLabel(self.tr('s1_url_lbl'))
        layout.addWidget(self.s1_lbl_url, 0, 0)
        self.s1_url = QLineEdit("https://platesmania.com/de/gallery.php?ctype=11&gal=de")
        layout.addWidget(self.s1_url, 0, 1, 1, 3)

        self.s1_lbl_start = QLabel(self.tr('s1_start_page'))
        layout.addWidget(self.s1_lbl_start, 1, 0)
        self.s1_start_page = QSpinBox()
        self.s1_start_page.setRange(1, 10000)
        self.s1_start_page.setValue(2)
        layout.addWidget(self.s1_start_page, 1, 1)

        self.s1_lbl_count = QLabel(self.tr('s1_page_count'))
        layout.addWidget(self.s1_lbl_count, 1, 2)
        self.s1_page_count = QSpinBox()
        self.s1_page_count.setRange(1, 10000)
        self.s1_page_count.setValue(100)
        layout.addWidget(self.s1_page_count, 1, 3)

        self.s1_lbl_wait = QLabel(self.tr('s1_wait_time'))
        layout.addWidget(self.s1_lbl_wait, 2, 0)
        self.s1_wait_time = QDoubleSpinBox()
        self.s1_wait_time.setRange(0.0, 60.0)
        self.s1_wait_time.setValue(2.0)
        layout.addWidget(self.s1_wait_time, 2, 1)

        self.s1_lbl_notes = QLabel(self.tr('s1_notes_file'))
        layout.addWidget(self.s1_lbl_notes, 2, 2)
        self.s1_notes_file = QLineEdit("plakalar4.txt")
        layout.addWidget(self.s1_notes_file, 2, 3)

        self.s1_lbl_state = QLabel(self.tr('s1_state_file'))
        layout.addWidget(self.s1_lbl_state, 3, 0)
        self.s1_state_file = QLineEdit("kayit.json")
        layout.addWidget(self.s1_state_file, 3, 1)

        self.s1_lbl_linksel = QLabel(self.tr('s1_link_sel'))
        layout.addWidget(self.s1_lbl_linksel, 3, 2)
        self.s1_link_sel = QLineEdit("a[href*='/de/nomer']")
        layout.addWidget(self.s1_link_sel, 3, 3)

        self.s1_lbl_imgsel = QLabel(self.tr('s1_img_sel'))
        layout.addWidget(self.s1_lbl_imgsel, 4, 0)
        self.s1_img_sel = QLineEdit("img.img-responsive.center-block")
        layout.addWidget(self.s1_img_sel, 4, 1, 1, 3)

        self.btn_s1_run = QPushButton(self.tr('s1_btn'))
        self.btn_s1_run.setObjectName("stepBtn")
        self.btn_s1_run.clicked.connect(lambda: self.start_pipeline([1]))
        layout.addWidget(self.btn_s1_run, 5, 0, 1, 4)

        self.tabs.addTab(tab, self.tr('tab1'))

    def setup_tab_2(self):
        tab = QWidget()
        layout = QGridLayout(tab)

        self.s2_lbl_model = QLabel(self.tr('s2_model_lbl'))
        layout.addWidget(self.s2_lbl_model, 0, 0)
        self.s2_model_path = QLineEdit("V2.pt")
        layout.addWidget(self.s2_model_path, 0, 1, 1, 2)
        self.btn_model = QPushButton(self.tr('s2_model_btn'))
        self.btn_model.clicked.connect(self.browse_model)
        layout.addWidget(self.btn_model, 0, 3)

        self.s2_lbl_class = QLabel(self.tr('s2_class_lbl'))
        layout.addWidget(self.s2_lbl_class, 1, 0)
        self.s2_class_id = QSpinBox()
        self.s2_class_id.setRange(0, 999)
        self.s2_class_id.setValue(1)
        layout.addWidget(self.s2_class_id, 1, 1)

        self.s2_lbl_conf = QLabel(self.tr('s2_conf_lbl'))
        layout.addWidget(self.s2_lbl_conf, 1, 2)
        self.s2_conf = QDoubleSpinBox()
        self.s2_conf.setRange(0.01, 1.0)
        self.s2_conf.setSingleStep(0.05)
        self.s2_conf.setValue(0.25)
        layout.addWidget(self.s2_conf, 1, 3)

        self.s2_lbl_imgsz = QLabel(self.tr('s2_imgsz_lbl'))
        layout.addWidget(self.s2_lbl_imgsz, 2, 0)
        self.s2_imgsz = QSpinBox()
        self.s2_imgsz.setRange(128, 2048)
        self.s2_imgsz.setSingleStep(32)
        self.s2_imgsz.setValue(640)
        layout.addWidget(self.s2_imgsz, 2, 1)

        self.btn_s2_run = QPushButton(self.tr('s2_btn'))
        self.btn_s2_run.setObjectName("stepBtn")
        self.btn_s2_run.clicked.connect(lambda: self.start_pipeline([2]))
        layout.addWidget(self.btn_s2_run, 3, 0, 1, 4)

        self.tabs.addTab(tab, self.tr('tab2'))

    def setup_tab_3(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.s3_info_lbl = QLabel(self.tr('s3_info'))
        self.s3_info_lbl.setStyleSheet("color: #cbd5e1; font-size: 13px; line-height: 1.5;")
        layout.addWidget(self.s3_info_lbl)

        self.btn_s3_run = QPushButton(self.tr('s3_btn'))
        self.btn_s3_run.setObjectName("stepBtn")
        self.btn_s3_run.clicked.connect(lambda: self.start_pipeline([3]))
        layout.addWidget(self.btn_s3_run)

        self.tabs.addTab(tab, self.tr('tab3'))

    def setup_tab_4(self):
        tab = QWidget()
        layout = QGridLayout(tab)

        self.s4_lbl_start = QLabel(self.tr('s4_start_no'))
        layout.addWidget(self.s4_lbl_start, 0, 0)
        self.s4_start_no = QSpinBox()
        self.s4_start_no.setRange(1, 999999)
        self.s4_start_no.setValue(1)
        layout.addWidget(self.s4_start_no, 0, 1)

        self.s4_lbl_exclude = QLabel(self.tr('s4_exclude'))
        layout.addWidget(self.s4_lbl_exclude, 1, 0)
        self.s4_exclude = QLineEdit("classes, notes")
        layout.addWidget(self.s4_exclude, 1, 1)

        self.s4_info_lbl = QLabel(self.tr('s4_info'))
        self.s4_info_lbl.setStyleSheet("color: #94a3b8; line-height: 1.4;")
        layout.addWidget(self.s4_info_lbl, 2, 0, 1, 2)

        self.btn_s4_run = QPushButton(self.tr('s4_btn'))
        self.btn_s4_run.setObjectName("stepBtn")
        self.btn_s4_run.clicked.connect(lambda: self.start_pipeline([4]))
        layout.addWidget(self.btn_s4_run, 3, 0, 1, 2)

        self.tabs.addTab(tab, self.tr('tab4'))

    def setup_tab_5(self):
        tab = QWidget()
        layout = QGridLayout(tab)

        self.s5_lbl_out = QLabel(self.tr('s5_out_dir'))
        layout.addWidget(self.s5_lbl_out, 0, 0)
        self.s5_output_dir = QLineEdit("czeck_dataset")
        layout.addWidget(self.s5_output_dir, 0, 1)

        self.s5_lbl_train = QLabel(self.tr('s5_train_ratio'))
        layout.addWidget(self.s5_lbl_train, 0, 2)
        self.s5_train_ratio = QDoubleSpinBox()
        self.s5_train_ratio.setRange(0.05, 0.95)
        self.s5_train_ratio.setSingleStep(0.05)
        self.s5_train_ratio.setValue(0.70)
        self.s5_train_ratio.valueChanged.connect(self.update_test_ratio_label)
        layout.addWidget(self.s5_train_ratio, 0, 3)

        self.s5_lbl_val = QLabel(self.tr('s5_val_ratio'))
        layout.addWidget(self.s5_lbl_val, 0, 4)
        self.s5_val_ratio = QDoubleSpinBox()
        self.s5_val_ratio.setRange(0.05, 0.95)
        self.s5_val_ratio.setSingleStep(0.05)
        self.s5_val_ratio.setValue(0.15)
        self.s5_val_ratio.valueChanged.connect(self.update_test_ratio_label)
        layout.addWidget(self.s5_val_ratio, 0, 5)

        self.s5_test_label = QLabel("Test Oranı: %15")
        self.s5_test_label.setStyleSheet("color: #38bdf8; font-weight: bold;")
        layout.addWidget(self.s5_test_label, 0, 6)

        self.s5_lbl_s1_img = QLabel(self.tr('s5_src1_img'))
        layout.addWidget(self.s5_lbl_s1_img, 1, 0)
        self.s5_src1_img = QLineEdit(self.raw_folder_edit.text())
        layout.addWidget(self.s5_src1_img, 1, 1, 1, 2)

        self.s5_lbl_s1_txt = QLabel(self.tr('s5_src1_txt'))
        layout.addWidget(self.s5_lbl_s1_txt, 1, 3)
        self.s5_src1_txt = QLineEdit(self.labels_folder_edit.text())
        layout.addWidget(self.s5_src1_txt, 1, 4, 1, 3)

        self.s5_lbl_s2_img = QLabel(self.tr('s5_src2_img'))
        layout.addWidget(self.s5_lbl_s2_img, 2, 0)
        self.s5_src2_img = QLineEdit("")
        self.s5_src2_img.setPlaceholderText("Örn: motorlar")
        layout.addWidget(self.s5_src2_img, 2, 1, 1, 2)

        self.s5_lbl_s2_txt = QLabel(self.tr('s5_src2_txt'))
        layout.addWidget(self.s5_lbl_s2_txt, 2, 3)
        self.s5_src2_txt = QLineEdit("")
        self.s5_src2_txt.setPlaceholderText("Örn: etiketler_moto")
        layout.addWidget(self.s5_src2_txt, 2, 4, 1, 3)

        self.btn_s5_run = QPushButton(self.tr('s5_btn'))
        self.btn_s5_run.setObjectName("stepBtn")
        self.btn_s5_run.clicked.connect(lambda: self.start_pipeline([5]))
        layout.addWidget(self.btn_s5_run, 3, 0, 1, 7)

        self.tabs.addTab(tab, self.tr('tab5'))

    def setup_tab_settings(self):
        tab = QWidget()
        layout = QGridLayout(tab)

        self.set_lbl_font = QLabel(self.tr('set_font_lbl'))
        layout.addWidget(self.set_lbl_font, 0, 0)
        self.font_spin = QSpinBox()
        self.font_spin.setRange(8, 20)
        self.font_spin.setValue(10)
        self.font_spin.valueChanged.connect(lambda v: self.log_output.setFont(QFont("Consolas", v)))
        layout.addWidget(self.font_spin, 0, 1)

        self.set_lbl_anim = QLabel(self.tr('set_anim_lbl'))
        layout.addWidget(self.set_lbl_anim, 1, 0)
        self.anim_combo = QComboBox()
        self.anim_combo.addItems(["Açık / Enabled", "Kapalı / Disabled"])
        self.anim_combo.currentIndexChanged.connect(lambda idx: setattr(self, 'animations_enabled', idx == 0))
        layout.addWidget(self.anim_combo, 1, 1)

        self.btn_save_cfg = QPushButton(self.tr('set_save_btn'))
        self.btn_save_cfg.clicked.connect(self.save_settings_json)
        layout.addWidget(self.btn_save_cfg, 2, 0)

        self.btn_load_cfg = QPushButton(self.tr('set_load_btn'))
        self.btn_load_cfg.clicked.connect(self.load_settings_json)
        layout.addWidget(self.btn_load_cfg, 2, 1)

        layout.setRowStretch(3, 1)
        self.tabs.addTab(tab, self.tr('tab6'))

    # --- ANİMASYON FONKSİYONLARI ---
    def animate_tab_transition(self, index):
        if not self.animations_enabled:
            return
        widget = self.tabs.widget(index)
        if not widget:
            return
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        self.tab_anim = QPropertyAnimation(effect, b"opacity")
        self.tab_anim.setDuration(220)
        self.tab_anim.setStartValue(0.15)
        self.tab_anim.setEndValue(1.0)
        self.tab_anim.setEasingCurve(QEasingCurve.OutCubic)
        self.tab_anim.finished.connect(lambda: widget.setGraphicsEffect(None))
        self.tab_anim.start()

    def toggle_console_animated(self):
        sizes = self.main_splitter.sizes()
        total = sum(sizes)
        if sizes[1] > 20:
            self.saved_console_height = sizes[1]
            start_h, end_h = sizes[1], 0
        else:
            start_h = 0
            end_h = getattr(self, 'saved_console_height', 260)

        if not self.animations_enabled:
            self.main_splitter.setSizes([total - end_h, end_h])
            return

        self.split_anim = QVariantAnimation(self)
        self.split_anim.setDuration(280)
        self.split_anim.setStartValue(start_h)
        self.split_anim.setEndValue(end_h)
        self.split_anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.split_anim.valueChanged.connect(lambda val: self.main_splitter.setSizes([total - int(val), int(val)]))
        self.split_anim.start()

    # --- AYARLAR VE DİL FONKSİYONLARI ---
    def save_settings_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Config", self.base_dir_edit.text(), "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.get_config(), f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "OK", "Ayarlar kaydedildi / Settings saved!")

    def load_settings_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", self.base_dir_edit.text(), "JSON (*.json)")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.base_dir_edit.setText(cfg.get('base_dir', self.base_dir_edit.text()))
            self.raw_folder_edit.setText(cfg.get('raw_folder', 'elektrikli_araclar'))
            self.drawn_folder_edit.setText(cfg.get('drawn_folder', 'label4'))
            self.labels_folder_edit.setText(cfg.get('labels_folder', 'etiketler4'))
            self.s1_url.setText(cfg.get('s1_url', ''))
            self.s1_notes_file.setText(cfg.get('s1_notes_file', 'plakalar4.txt'))
            self.s2_model_path.setText(cfg.get('s2_model_path', 'V2.pt'))
            QMessageBox.information(self, "OK", "Ayarlar yüklendi / Settings loaded!")

    def change_language(self, index):
        self.current_lang = 'tr' if index == 0 else 'en'
        self.setWindowTitle(self.tr('window_title'))
        self.top_group.setTitle(self.tr('common_dir'))
        self.lbl_base_dir.setText(self.tr('base_dir_lbl'))
        self.btn_browse_base.setText(self.tr('browse_btn'))
        self.lbl_raw_folder.setText(self.tr('raw_folder_lbl'))
        self.lbl_drawn_folder.setText(self.tr('drawn_folder_lbl'))
        self.lbl_labels_folder.setText(self.tr('labels_folder_lbl'))
        self.btn_run_all.setText(self.tr('run_all_btn'))
        self.btn_continue.setText(self.tr('continue_btn'))
        self.log_group.setTitle(self.tr('log_group'))
        self.btn_clear_log.setText(self.tr('clear_log'))
        self.btn_toggle_console.setText(self.tr('toggle_log'))

        for i, key in enumerate(['tab1', 'tab2', 'tab3', 'tab4', 'tab5', 'tab6']):
            self.tabs.setTabText(i, self.tr(key))

        self.s1_lbl_url.setText(self.tr('s1_url_lbl'))
        self.s1_lbl_start.setText(self.tr('s1_start_page'))
        self.s1_lbl_count.setText(self.tr('s1_page_count'))
        self.s1_lbl_wait.setText(self.tr('s1_wait_time'))
        self.s1_lbl_notes.setText(self.tr('s1_notes_file'))
        self.s1_lbl_state.setText(self.tr('s1_state_file'))
        self.s1_lbl_linksel.setText(self.tr('s1_link_sel'))
        self.s1_lbl_imgsel.setText(self.tr('s1_img_sel'))
        self.btn_s1_run.setText(self.tr('s1_btn'))

        self.s2_lbl_model.setText(self.tr('s2_model_lbl'))
        self.btn_model.setText(self.tr('s2_model_btn'))
        self.s2_lbl_class.setText(self.tr('s2_class_lbl'))
        self.s2_lbl_conf.setText(self.tr('s2_conf_lbl'))
        self.s2_lbl_imgsz.setText(self.tr('s2_imgsz_lbl'))
        self.btn_s2_run.setText(self.tr('s2_btn'))

        self.s3_info_lbl.setText(self.tr('s3_info'))
        self.btn_s3_run.setText(self.tr('s3_btn'))

        self.s4_lbl_start.setText(self.tr('s4_start_no'))
        self.s4_lbl_exclude.setText(self.tr('s4_exclude'))
        self.s4_info_lbl.setText(self.tr('s4_info'))
        self.btn_s4_run.setText(self.tr('s4_btn'))

        self.s5_lbl_out.setText(self.tr('s5_out_dir'))
        self.s5_lbl_train.setText(self.tr('s5_train_ratio'))
        self.s5_lbl_val.setText(self.tr('s5_val_ratio'))
        self.s5_lbl_s1_img.setText(self.tr('s5_src1_img'))
        self.s5_lbl_s1_txt.setText(self.tr('s5_src1_txt'))
        self.s5_lbl_s2_img.setText(self.tr('s5_src2_img'))
        self.s5_lbl_s2_txt.setText(self.tr('s5_src2_txt'))
        self.btn_s5_run.setText(self.tr('s5_btn'))

        self.set_lbl_font.setText(self.tr('set_font_lbl'))
        self.set_lbl_anim.setText(self.tr('set_anim_lbl'))
        self.btn_save_cfg.setText(self.tr('set_save_btn'))
        self.btn_load_cfg.setText(self.tr('set_load_btn'))

    def sync_step5_primary_source(self):
        if hasattr(self, 's5_src1_img'):
            self.s5_src1_img.setText(self.raw_folder_edit.text())
        if hasattr(self, 's5_src1_txt'):
            self.s5_src1_txt.setText(self.labels_folder_edit.text())

    def update_test_ratio_label(self):
        rem = 1.0 - (self.s5_train_ratio.value() + self.s5_val_ratio.value())
        self.s5_test_label.setText(f"Test Oranı / Ratio: %{int(round(rem * 100))}")

    def browse_base_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory", self.base_dir_edit.text())
        if folder:
            self.base_dir_edit.setText(folder)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", self.base_dir_edit.text(), "YOLO Model (*.pt)"
        )
        if file_path:
            self.s2_model_path.setText(file_path)

    def get_config(self):
        return {
            'base_dir': self.base_dir_edit.text().strip(),
            'raw_folder': self.raw_folder_edit.text().strip(),
            'drawn_folder': self.drawn_folder_edit.text().strip(),
            'labels_folder': self.labels_folder_edit.text().strip(),

            's1_url': self.s1_url.text().strip(),
            's1_start_page': self.s1_start_page.value(),
            's1_page_count': self.s1_page_count.value(),
            's1_wait_time': self.s1_wait_time.value(),
            's1_notes_file': self.s1_notes_file.text().strip(),
            's1_state_file': self.s1_state_file.text().strip(),
            's1_link_selector': self.s1_link_sel.text().strip(),
            's1_img_selector': self.s1_img_sel.text().strip(),

            's2_model_path': self.s2_model_path.text().strip(),
            's2_class_id': self.s2_class_id.value(),
            's2_conf': self.s2_conf.value(),
            's2_imgsz': self.s2_imgsz.value(),

            's4_start_no': self.s4_start_no.value(),
            's4_exclude': self.s4_exclude.text().strip(),

            's5_output_dir': self.s5_output_dir.text().strip(),
            's5_train_ratio': self.s5_train_ratio.value(),
            's5_val_ratio': self.s5_val_ratio.value(),
            's5_sources': [
                (self.s5_src1_img.text(), self.s5_src1_txt.text()),
                (self.s5_src2_img.text(), self.s5_src2_txt.text())
            ]
        }

    def start_pipeline(self, steps):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Uyarı / Warning", "Şu anda çalışan bir işlem var! / A task is already running!")
            return

        cfg = self.get_config()
        if 5 in steps and (cfg['s5_train_ratio'] + cfg['s5_val_ratio'] >= 1.0):
            QMessageBox.warning(self, "Hatalı Oran", "Train + Val oranı 1.0'dan küçük olmalıdır!")
            return

        self.btn_run_all.setEnabled(False)
        self.worker = PipelineWorker(steps, cfg)
        self.worker.log_signal.connect(self.append_log)
        self.worker.status_signal.connect(lambda s: self.status_label.setText(s))
        self.worker.wait_input_signal.connect(self.on_wait_user_input)
        self.worker.finished_signal.connect(self.on_pipeline_finished)
        self.worker.start()

    def append_log(self, text):
        self.log_output.append(text)
        self.log_output.moveCursor(QTextCursor.End)

    def on_wait_user_input(self, msg):
        self.btn_continue.setEnabled(True)
        self.btn_continue.setStyleSheet("background-color: #facc15; color: #0f172a; font-weight: bold; border-radius: 8px;")
        self.status_label.setText(self.tr('status_wait'))

    def on_continue_clicked(self):
        if self.worker:
            self.btn_continue.setEnabled(False)
            self.btn_continue.setStyleSheet("")
            self.worker.resume_from_input()

    def on_pipeline_finished(self, success, message):
        self.btn_run_all.setEnabled(True)
        self.btn_continue.setEnabled(False)
        if success:
            QMessageBox.information(self, "Tamamlandı / Completed", message)
        else:
            QMessageBox.critical(self, "Hata / Error", f"Hata meydana geldi:\n{message}")

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0b0f19;
                color: #f1f5f9;
                font-family: 'Segoe UI', -apple-system, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #1e293b;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 16px;
                font-weight: bold;
                color: #38bdf8;
                background-color: #0f172a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 6px 12px;
                color: #f8fafc;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #38bdf8;
            }
            QTabWidget::pane {
                border: 1px solid #1e293b;
                border-radius: 10px;
                background-color: #0f172a;
                padding: 14px;
            }
            QTabBar::tab {
                background-color: #0b0f19;
                border: 1px solid #1e293b;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 10px 16px;
                margin-right: 4px;
                color: #64748b;
            }
            QTabBar::tab:selected {
                background-color: #0f172a;
                color: #38bdf8;
                font-weight: bold;
                border-top: 2px solid #38bdf8;
            }
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 8px 16px;
                color: #f8fafc;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #334155;
                border-color: #475569;
            }
            QPushButton:pressed {
                background-color: #0f172a;
            }
            QPushButton:disabled {
                background-color: #0f172a;
                color: #475569;
                border-color: #1e293b;
            }
            QPushButton#stepBtn {
                background-color: #0284c7;
                border: none;
                padding: 12px;
                font-size: 14px;
                margin-top: 10px;
                border-radius: 8px;
            }
            QPushButton#stepBtn:hover {
                background-color: #0369a1;
            }
            QPushButton#runAllBtn {
                background-color: #16a34a;
                border: none;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton#runAllBtn:hover {
                background-color: #15803d;
            }
            QTextEdit {
                background-color: #020617;
                border: 1px solid #1e293b;
                border-radius: 8px;
                color: #34d399;
                padding: 10px;
            }
            QFrame#actionFrame {
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 10px;
                padding: 6px;
            }
            QSplitter::handle {
                background-color: #334155;
                height: 5px;
                border-radius: 2px;
                margin: 2px 40px;
            }
            QSplitter::handle:hover {
                background-color: #38bdf8;
            }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())