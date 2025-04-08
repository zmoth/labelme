@echo off
set LABELME_PATH=%CD%\labelme

for /f "delims=" %%i in ('python -c "import os, osam; print(os.path.dirname(osam.__file__).replace('/', '\\'))"') do set OSAM_PATH=%%i
for /f "delims=" %%i in ('python -c "import os, pylibdmtx; print(os.path.dirname(pylibdmtx.__file__).replace('/', '\\'))"') do set DMTX_PATH=%%i

pyinstaller labelme\__main__.py ^
  --name=Labelme ^
  --windowed ^
  --noconfirm ^
  --specpath=build ^
  --add-data="%OSAM_PATH%\_models\yoloworld\clip\bpe_simple_vocab_16e6.txt.gz;osam\_models\yoloworld\clip" ^
  --add-data="%DMTX_PATH%\*.dll;pylibdmtx" ^
  --add-data="%LABELME_PATH%\config\default_config.yaml;labelme\config" ^
  --add-data="%LABELME_PATH%\icons\*;labelme\icons" ^
  --add-data="%LABELME_PATH%\translate\*;translate" ^
  --icon="%LABELME_PATH%\icons\icon.png" ^
  --onefile

copy /Y "C:\Windows\System32\msvcp140.dll" "%CD%\dist"