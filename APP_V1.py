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
    QTabWidget, QTextEdit, QFileDialog, QGroupBox, QMessageBox, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, QMutex, QWaitCondition
from PySide6.QtGui import QFont, QTextCursor

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


# ==============================================================================
# ARKA PLAN İŞÇİ (THREAD) SINIFI - ARAYÜZÜN DONMASINI ENGELLER VE SIRALI ÇALIŞTIRIR
# ==============================================================================
class PipelineWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(str)
    wait_input_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, steps, config):
        super().__init__()
        self.steps = steps  # Örn: [1, 2, 3, 4, 5] veya sadece [2]
        self.cfg = config
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.is_cancelled = False

    def log(self, text):
        self.log_signal.emit(str(text))

    def wait_for_user(self, prompt_msg):
        """1. Scriptteki input() yerine arayüzden buton onayı bekler."""
        self.log(f"\n[BEKLENİYOR] {prompt_msg}")
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
                    self.status_signal.emit("Adım 1/5: Görseller İndiriliyor...")
                    self.run_step_1()
                elif step == 2:
                    self.status_signal.emit("Adım 2/5: YOLO ile Etiketleniyor...")
                    self.run_step_2()
                elif step == 3:
                    self.status_signal.emit("Adım 3/5: Hatalı Etiketler Temizleniyor...")
                    self.run_step_3()
                elif step == 4:
                    self.status_signal.emit("Adım 4/5: Dosyalar Yeniden Adlandırılıyor...")
                    self.run_step_4()
                elif step == 5:
                    self.status_signal.emit("Adım 5/5: Train / Val / Test Ayrılıyor...")
                    self.run_step_5()

            self.status_signal.emit("Hazır")
            self.finished_signal.emit(True, "Tüm seçili işlemler başarıyla tamamlandı!")
        except Exception as e:
            self.log(f"\n[KRİTİK HATA]: {str(e)}")
            self.status_signal.emit("Hata Oluştu!")
            self.finished_signal.emit(False, str(e))

    # --------------------------------------------------------------------------
    # 1. SCRIPT: VERİ KAZIMA VE GÖRSEL İNDİRME
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # 2. SCRIPT: YOLO İLE LABEL OLUŞTURMA
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # 3. SCRIPT: HATALI LABEL KONTROL EDİCİ VE SİLİCİ
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # 4. SCRIPT: EŞZAMANLI AD DEĞİŞTİRİCİ (TEMP İLE GÜVENLİ)
    # --------------------------------------------------------------------------
    def run_step_4(self):
        self.log("\n" + "=" * 60)
        self.log(">>> 4. ADIM BAŞLADI: Eşzamanlı Yeniden Adlandırma")
        self.log("=" * 60)

        ANA_DIZIN = self.cfg['base_dir']
        klasor_gorsel_1 = os.path.join(ANA_DIZIN, self.cfg['raw_folder'])
        klasor_gorsel_2 = os.path.join(ANA_DIZIN, self.cfg['drawn_folder'])
        klasor_txt = os.path.join(ANA_DIZIN, self.cfg['labels_folder'])

        klasor_listesi = [klasor_gorsel_1, klasor_gorsel_2, klasor_txt]
        haric_tutulacaklar = {x.strip() for x in self.cfg['s4_exclude'].split(",") if x.strip()}
        baslangic_no = self.cfg['s4_start_no']

        benzersiz_isimler = set()

        for klasor in klasor_listesi:
            if not os.path.exists(klasor):
                raise FileNotFoundError(f"[HATA] Klasör bulunamadı: {klasor}")

            for dosya in os.listdir(klasor):
                tam_yol = os.path.join(klasor, dosya)
                if os.path.isfile(tam_yol):
                    isim, _ = os.path.splitext(dosya)
                    if isim not in haric_tutulacaklar:
                        benzersiz_isimler.add(isim)

        sirali_isimler = sorted(list(benzersiz_isimler))
        isim_haritasi = {eski_isim: f"img({i})" for i, eski_isim in enumerate(sirali_isimler, start=baslangic_no)}

        self.log(f"Toplam {len(isim_haritasi)} adet benzersiz dosya grubu bulundu. İşlem başlıyor...\n")

        gecici_dosyalar = []

        # Çakışmaları önlemek için önce temp isimlere çevir
        for klasor in klasor_listesi:
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

        # Geçici isimleri nihai img(x) isimlerine çevir
        for gecici_yol, hedef_yol, eski_ad, yeni_ad in gecici_dosyalar:
            os.rename(gecici_yol, hedef_yol)

        self.log("İşlem başarıyla tamamlandı! Örnek eşleşmeler:")
        for eski, yeni in list(isim_haritasi.items())[:10]:
            self.log(f"  '{eski}' -> '{yeni}'")

    # --------------------------------------------------------------------------
    # 5. SCRIPT: TRAIN / VAL / TEST AYIRMA İŞLEMİ
    # --------------------------------------------------------------------------
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


# ==============================================================================
# ANA ARAYÜZ PENCERESİ (MODERN PYSIDE6 GUI)
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Veri Seti Otomasyon Stüdyosu")
        self.setMinimumSize(980, 760)
        self.worker = None
        self.setup_ui()
        self.apply_stylesheet()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # 1. ÜST PANEL: ORTAK KLASÖR VE DİZİN AYARLARI
        top_group = QGroupBox("Ortak Çalışma Dizini ve Klasör İsimleri (1, 2, 3 ve 4. Adımlar İçin)")
        top_layout = QGridLayout(top_group)

        top_layout.addWidget(QLabel("Ana Çalışma Dizini:"), 0, 0)
        self.base_dir_edit = QLineEdit(os.path.dirname(os.path.abspath(__file__)))
        top_layout.addWidget(self.base_dir_edit, 0, 1, 1, 4)
        btn_browse_base = QPushButton("Klasör Seç")
        btn_browse_base.clicked.connect(self.browse_base_dir)
        top_layout.addWidget(btn_browse_base, 0, 5)

        top_layout.addWidget(QLabel("Ham Görsel Klasörü:"), 1, 0)
        self.raw_folder_edit = QLineEdit("elektrikli_araclar")
        self.raw_folder_edit.textChanged.connect(self.sync_step5_primary_source)
        top_layout.addWidget(self.raw_folder_edit, 1, 1)

        top_layout.addWidget(QLabel("Çizimli Görsel Klasörü:"), 1, 2)
        self.drawn_folder_edit = QLineEdit("label4")
        top_layout.addWidget(self.drawn_folder_edit, 1, 3)

        top_layout.addWidget(QLabel("TXT Etiket Klasörü:"), 1, 4)
        self.labels_folder_edit = QLineEdit("etiketler4")
        self.labels_folder_edit.textChanged.connect(self.sync_step5_primary_source)
        top_layout.addWidget(self.labels_folder_edit, 1, 5)

        main_layout.addWidget(top_group)

        # 2. ORTA PANEL: SEKMELİ SCRIPT AYARLARI
        self.tabs = QTabWidget()
        self.setup_tab_1()
        self.setup_tab_2()
        self.setup_tab_3()
        self.setup_tab_4()
        self.setup_tab_5()
        main_layout.addWidget(self.tabs)

        # 3. AKSİYON BAR: TÜMÜNÜ ÇALIŞTIR VE ETKİLEŞİM BUTONU
        action_frame = QFrame()
        action_frame.setObjectName("actionFrame")
        action_layout = QHBoxLayout(action_frame)

        self.btn_run_all = QPushButton("🚀 TÜMÜNÜ SIRAYLA ÇALIŞTIR (1 ➔ 2 ➔ 3 ➔ 4 ➔ 5)")
        self.btn_run_all.setObjectName("runAllBtn")
        self.btn_run_all.setFixedHeight(42)
        self.btn_run_all.clicked.connect(lambda: self.start_pipeline([1, 2, 3, 4, 5]))
        action_layout.addWidget(self.btn_run_all, stretch=2)

        self.btn_continue = QPushButton("⏸️ Devam Et / Onayla (Cloudflare & Tarayıcı)")
        self.btn_continue.setObjectName("continueBtn")
        self.btn_continue.setFixedHeight(42)
        self.btn_continue.setEnabled(False)
        self.btn_continue.clicked.connect(self.on_continue_clicked)
        action_layout.addWidget(self.btn_continue, stretch=1)

        self.status_label = QLabel("Durum: Hazır")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_label.setStyleSheet("font-weight: bold; color: #38bdf8; padding-right: 8px;")
        action_layout.addWidget(self.status_label, stretch=1)

        main_layout.addWidget(action_frame)

        # 4. ALT PANEL: CANLI LOG EKRANI
        log_group = QGroupBox("Canlı İşlem Çıktısı (Konsol / Log)")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 10))
        log_layout.addWidget(self.log_output)

        btn_clear_log = QPushButton("Logları Temizle")
        btn_clear_log.setFixedWidth(140)
        btn_clear_log.clicked.connect(self.log_output.clear)
        log_layout.addWidget(btn_clear_log, alignment=Qt.AlignRight)

        main_layout.addWidget(log_group, stretch=1)

    # ---------------- SEKME 1: GÖRSEL İNDİRİCİ ----------------
    def setup_tab_1(self):
        tab = QWidget()
        layout = QGridLayout(tab)

        layout.addWidget(QLabel("Hedef Galeri URL:"), 0, 0)
        self.s1_url = QLineEdit("https://platesmania.com/de/gallery.php?ctype=11&gal=de")
        layout.addWidget(self.s1_url, 0, 1, 1, 3)

        layout.addWidget(QLabel("Başlangıç Sayfası:"), 1, 0)
        self.s1_start_page = QSpinBox()
        self.s1_start_page.setRange(1, 10000)
        self.s1_start_page.setValue(2)
        layout.addWidget(self.s1_start_page, 1, 1)

        layout.addWidget(QLabel("Taranacak Sayfa Sayısı:"), 1, 2)
        self.s1_page_count = QSpinBox()
        self.s1_page_count.setRange(1, 10000)
        self.s1_page_count.setValue(100)
        layout.addWidget(self.s1_page_count, 1, 3)

        layout.addWidget(QLabel("Bekleme Süresi (sn):"), 2, 0)
        self.s1_wait_time = QDoubleSpinBox()
        self.s1_wait_time.setRange(0.0, 60.0)
        self.s1_wait_time.setValue(2.0)
        layout.addWidget(self.s1_wait_time, 2, 1)

        layout.addWidget(QLabel("Plaka Not Dosyası (.txt):"), 2, 2)
        self.s1_notes_file = QLineEdit("plakalar4.txt")
        layout.addWidget(self.s1_notes_file, 2, 3)

        layout.addWidget(QLabel("Durum Dosyası (.json):"), 3, 0)
        self.s1_state_file = QLineEdit("kayit.json")
        layout.addWidget(self.s1_state_file, 3, 1)

        layout.addWidget(QLabel("Link Seçici (CSS):"), 3, 2)
        self.s1_link_sel = QLineEdit("a[href*='/de/nomer']")
        layout.addWidget(self.s1_link_sel, 3, 3)

        layout.addWidget(QLabel("Görsel Seçici (CSS):"), 4, 0)
        self.s1_img_sel = QLineEdit("img.img-responsive.center-block")
        layout.addWidget(self.s1_img_sel, 4, 1, 1, 3)

        btn_run = QPushButton("▶ Sadece 1. Scripti (Görsel İndirici) Çalıştır")
        btn_run.setObjectName("stepBtn")
        btn_run.clicked.connect(lambda: self.start_pipeline([1]))
        layout.addWidget(btn_run, 5, 0, 1, 4)

        self.tabs.addTab(tab, "1. Görsel İndirici")

    # ---------------- SEKME 2: YOLO ETİKETLEME ----------------
    def setup_tab_2(self):
        tab = QWidget()
        layout = QGridLayout(tab)

        layout.addWidget(QLabel("YOLO Model Dosyası (.pt):"), 0, 0)
        self.s2_model_path = QLineEdit("V2.pt")
        layout.addWidget(self.s2_model_path, 0, 1, 1, 2)
        btn_model = QPushButton("Model Seç")
        btn_model.clicked.connect(self.browse_model)
        layout.addWidget(btn_model, 0, 3)

        layout.addWidget(QLabel("Atanacak Class ID (Sınıf No):"), 1, 0)
        self.s2_class_id = QSpinBox()
        self.s2_class_id.setRange(0, 999)
        self.s2_class_id.setValue(1)
        layout.addWidget(self.s2_class_id, 1, 1)

        layout.addWidget(QLabel("Confidence (Güven Eşiği):"), 1, 2)
        self.s2_conf = QDoubleSpinBox()
        self.s2_conf.setRange(0.01, 1.0)
        self.s2_conf.setSingleStep(0.05)
        self.s2_conf.setValue(0.25)
        layout.addWidget(self.s2_conf, 1, 3)

        layout.addWidget(QLabel("Görsel Boyutu (imgsz):"), 2, 0)
        self.s2_imgsz = QSpinBox()
        self.s2_imgsz.setRange(128, 2048)
        self.s2_imgsz.setSingleStep(32)
        self.s2_imgsz.setValue(640)
        layout.addWidget(self.s2_imgsz, 2, 1)

        btn_run = QPushButton("▶ Sadece 2. Scripti (YOLO Etiketleme) Çalıştır")
        btn_run.setObjectName("stepBtn")
        btn_run.clicked.connect(lambda: self.start_pipeline([2]))
        layout.addWidget(btn_run, 3, 0, 1, 4)

        self.tabs.addTab(tab, "2. YOLO Etiketleme")

    # ---------------- SEKME 3: HATALI LABEL KONTROLÜ ----------------
    def setup_tab_3(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        info = QLabel(
            "Bu adım, üst panelde belirttiğiniz Ham Görsel, Çizimli Görsel ve TXT Etiket klasörlerini tarar.\n"
            "• Etiket dosyası boş olan (0 tespit) veya hiç bulunmayan görselleri,\n"
            "• İçinde 2 veya daha fazla etiket (Çoklu Tespit) bulunan görselleri\n"
            "her 3 klasörden de kalıcı olarak siler."
        )
        info.setStyleSheet("color: #cbd5e1; font-size: 13px; line-height: 1.5;")
        layout.addWidget(info)

        btn_run = QPushButton("▶ Sadece 3. Scripti (Hatalı Etiket Silici) Çalıştır")
        btn_run.setObjectName("stepBtn")
        btn_run.clicked.connect(lambda: self.start_pipeline([3]))
        layout.addWidget(btn_run)

        self.tabs.addTab(tab, "3. Hatalı Etiket Silici")

    # ---------------- SEKME 4: YENİDEN ADLANDIRICI ----------------
    def setup_tab_4(self):
        tab = QWidget()
        layout = QGridLayout(tab)

        layout.addWidget(QLabel("Başlangıç Numarası (Örn: img(1) için 1):"), 0, 0)
        self.s4_start_no = QSpinBox()
        self.s4_start_no.setRange(1, 999999)
        self.s4_start_no.setValue(1)
        layout.addWidget(self.s4_start_no, 0, 1)

        layout.addWidget(QLabel("Hariç Tutulacak Dosyalar (Virgülle):"), 1, 0)
        self.s4_exclude = QLineEdit("classes, notes")
        layout.addWidget(self.s4_exclude, 1, 1)

        info = QLabel("Not: Dosyalar çakışmayı önlemek için önce geçici (__temp_uuid) adlara, ardından img(N) formatına çevrilir.")
        info.setStyleSheet("color: #94a3b8;")
        layout.addWidget(info, 2, 0, 1, 2)

        btn_run = QPushButton("▶ Sadece 4. Scripti (Yeniden Adlandırıcı) Çalıştır")
        btn_run.setObjectName("stepBtn")
        btn_run.clicked.connect(lambda: self.start_pipeline([4]))
        layout.addWidget(btn_run, 3, 0, 1, 2)

        self.tabs.addTab(tab, "4. Ad Değiştirici")

    # ---------------- SEKME 5: TRAIN / VAL / TEST AYIRICI ----------------
    def setup_tab_5(self):
        tab = QWidget()
        layout = QGridLayout(tab)

        layout.addWidget(QLabel("Çıktı Dataset Klasör Adı:"), 0, 0)
        self.s5_output_dir = QLineEdit("czeck_dataset")
        layout.addWidget(self.s5_output_dir, 0, 1)

        layout.addWidget(QLabel("Train Oranı (Örn: 0.70):"), 0, 2)
        self.s5_train_ratio = QDoubleSpinBox()
        self.s5_train_ratio.setRange(0.05, 0.95)
        self.s5_train_ratio.setSingleStep(0.05)
        self.s5_train_ratio.setValue(0.70)
        self.s5_train_ratio.valueChanged.connect(self.update_test_ratio_label)
        layout.addWidget(self.s5_train_ratio, 0, 3)

        layout.addWidget(QLabel("Val Oranı (Örn: 0.15):"), 0, 4)
        self.s5_val_ratio = QDoubleSpinBox()
        self.s5_val_ratio.setRange(0.05, 0.95)
        self.s5_val_ratio.setSingleStep(0.05)
        self.s5_val_ratio.setValue(0.15)
        self.s5_val_ratio.valueChanged.connect(self.update_test_ratio_label)
        layout.addWidget(self.s5_val_ratio, 0, 5)

        self.s5_test_label = QLabel("Test Oranı: %15")
        self.s5_test_label.setStyleSheet("color: #38bdf8; font-weight: bold;")
        layout.addWidget(self.s5_test_label, 0, 6)

        # Kaynak Çiftleri (Birden fazla klasörü birleştirip bölebilmek için)
        layout.addWidget(QLabel("1. Kaynak Resim Klasörü:"), 1, 0)
        self.s5_src1_img = QLineEdit(self.raw_folder_edit.text())
        layout.addWidget(self.s5_src1_img, 1, 1, 1, 2)

        layout.addWidget(QLabel("1. Kaynak TXT Klasörü:"), 1, 3)
        self.s5_src1_txt = QLineEdit(self.labels_folder_edit.text())
        layout.addWidget(self.s5_src1_txt, 1, 4, 1, 3)

        layout.addWidget(QLabel("2. Kaynak Resim (Opsiyonel):"), 2, 0)
        self.s5_src2_img = QLineEdit("")
        self.s5_src2_img.setPlaceholderText("Örn: motorlar (Boş bırakılabilir)")
        layout.addWidget(self.s5_src2_img, 2, 1, 1, 2)

        layout.addWidget(QLabel("2. Kaynak TXT (Opsiyonel):"), 2, 3)
        self.s5_src2_txt = QLineEdit("")
        self.s5_src2_txt.setPlaceholderText("Örn: etiketler_moto (Boş bırakılabilir)")
        layout.addWidget(self.s5_src2_txt, 2, 4, 1, 3)

        btn_run = QPushButton("▶ Sadece 5. Scripti (Train/Val/Test Ayırıcı) Çalıştır")
        btn_run.setObjectName("stepBtn")
        btn_run.clicked.connect(lambda: self.start_pipeline([5]))
        layout.addWidget(btn_run, 3, 0, 1, 7)

        self.tabs.addTab(tab, "5. Dataset Ayırıcı (Train/Val/Test)")

    # ---------------- YARDIMCI FONKSİYONLAR ----------------
    def sync_step5_primary_source(self):
        """Üstteki klasör adları değiştikçe 5. sekmedeki 1. kaynak kutularını otomatik günceller."""
        if hasattr(self, 's5_src1_img'):
            self.s5_src1_img.setText(self.raw_folder_edit.text())
        if hasattr(self, 's5_src1_txt'):
            self.s5_src1_txt.setText(self.labels_folder_edit.text())

    def update_test_ratio_label(self):
        rem = 1.0 - (self.s5_train_ratio.value() + self.s5_val_ratio.value())
        self.s5_test_label.setText(f"Test Oranı: %{int(round(rem * 100))}")

    def browse_base_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Ana Çalışma Dizinini Seç", self.base_dir_edit.text())
        if folder:
            self.base_dir_edit.setText(folder)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "YOLO Model Dosyası Seç", self.base_dir_edit.text(), "YOLO Model (*.pt)"
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
            QMessageBox.warning(self, "İşlem Sürüyor", "Şu anda çalışan bir işlem var. Lütfen bitmesini bekleyin.")
            return

        cfg = self.get_config()
        if 5 in steps and (cfg['s5_train_ratio'] + cfg['s5_val_ratio'] >= 1.0):
            QMessageBox.warning(self, "Hatalı Oran", "Train ve Val oranlarının toplamı 1.0'dan küçük olmalıdır!")
            return

        self.btn_run_all.setEnabled(False)
        self.worker = PipelineWorker(steps, cfg)
        self.worker.log_signal.connect(self.append_log)
        self.worker.status_signal.connect(lambda s: self.status_label.setText(f"Durum: {s}"))
        self.worker.wait_input_signal.connect(self.on_wait_user_input)
        self.worker.finished_signal.connect(self.on_pipeline_finished)
        self.worker.start()

    def append_log(self, text):
        self.log_output.append(text)
        self.log_output.moveCursor(QTextCursor.End)

    def on_wait_user_input(self, msg):
        self.btn_continue.setEnabled(True)
        self.btn_continue.setStyleSheet("background-color: #eab308; color: #0f172a; font-weight: bold;")
        self.status_label.setText("Durum: Kullanıcı Onayı Bekleniyor...")

    def on_continue_clicked(self):
        if self.worker:
            self.btn_continue.setEnabled(False)
            self.btn_continue.setStyleSheet("")
            self.worker.resume_from_input()

    def on_pipeline_finished(self, success, message):
        self.btn_run_all.setEnabled(True)
        self.btn_continue.setEnabled(False)
        if success:
            QMessageBox.information(self, "Tamamlandı", message)
        else:
            QMessageBox.critical(self, "Hata Oluştu", f"İşlem sırasında hata meydana geldi:\n{message}")

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0f172a;
                color: #f8fafc;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 14px;
                font-weight: bold;
                color: #38bdf8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #1e293b;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 6px 10px;
                color: #f8fafc;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #38bdf8;
            }
            QTabWidget::pane {
                border: 1px solid #334155;
                border-radius: 8px;
                background-color: #1e293b;
                padding: 12px;
            }
            QTabBar::tab {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 8px 16px;
                margin-right: 4px;
                color: #94a3b8;
            }
            QTabBar::tab:selected {
                background-color: #1e293b;
                color: #38bdf8;
                font-weight: bold;
            }
            QPushButton {
                background-color: #334155;
                border: none;
                border-radius: 6px;
                padding: 8px 14px;
                color: #f8fafc;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #475569;
            }
            QPushButton:disabled {
                background-color: #1e293b;
                color: #64748b;
            }
            QPushButton#stepBtn {
                background-color: #0284c7;
                padding: 10px;
                font-size: 14px;
                margin-top: 8px;
            }
            QPushButton#stepBtn:hover {
                background-color: #0369a1;
            }
            QPushButton#runAllBtn {
                background-color: #16a34a;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton#runAllBtn:hover {
                background-color: #15803d;
            }
            QTextEdit {
                background-color: #020617;
                border: 1px solid #1e293b;
                border-radius: 6px;
                color: #a7f3d0;
                padding: 8px;
            }
            QFrame#actionFrame {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 4px;
            }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())