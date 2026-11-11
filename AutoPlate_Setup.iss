[Setup]
AppId={{A8F9E2D1-4B3C-4E5F-9A1B-2C3D4E5F6A7B}
AppName=AutoPlate
AppVersion=3.0
AppPublisher=OPTIMA
DefaultDirName={autopf}\AutoPlate
DefaultGroupName=AutoPlate
; Kurulum dosyasının kendi ikonu:
SetupIconFile=logo.ico
; Program Ekle/Kaldır'da görünecek ikon:
UninstallDisplayIcon={app}\AutoPlate.exe
; Setup dosyasının çıkacağı klasör ve adı:
OutputDir=.\Installer_Output
OutputBaseFilename=AutoPlate_Setup_v3.0
; Yüksek sıkıştırma ve hızlı kurulum optimizasyonu:
Compression=lzma2/fast
SolidCompression=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Yönetici izni gerektirmeden veya standart kurulumla yazma izni sorunu yaşamamak için:
PrivilegesRequired=lowest

[Languages]
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; 1. PyInstaller'ın oluşturduğu dist\AutoPlate klasöründeki TÜM dosyaları ve alt klasörleri (_internal) dahil et:
Source: "dist\AutoPlate\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; 2. V2.pt YOLO Modelini doğrudan kurulum klasörüne (AutoPlate.exe'nin yanına) kur:
Source: "V2.pt"; DestDir: "{app}"; Flags: ignoreversion

; 3. Logo dosyasını kurulum klasörüne kopyala:
Source: "logo.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\AutoPlate"; Filename: "{app}\AutoPlate.exe"; WorkingDir: "{app}"; IconFilename: "{app}\logo.ico"
Name: "{autodesktop}\AutoPlate"; Filename: "{app}\AutoPlate.exe"; WorkingDir: "{app}"; IconFilename: "{app}\logo.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\AutoPlate.exe"; Description: "{cm:LaunchProgram,AutoPlate}"; Flags: nowait postinstall skipifsilent