@echo off
chcp 65001 > NUL
rem set TORCH_HOME=C:\X\torch
rem set TORCH_EXTENSIONS_DIR=src\torch_utils\ops\.cache

if "%1"=="" goto help

if "%2"=="" goto test
if "%2"=="1" goto test

set model=%1
set name=%~n1
set size=%2
set frames=%3
set args=%4 %5 %6 %7 %8 %9
for %%q in (1 2 3 4 5 6 7 8 9 10) do shift 
set args=%args% %0 %1 %2 %3 %4 %5 %6 %7 %8 %9
for %%q in (1 2 3 4 5 6 7 8 9 10) do shift 
set args=%args% %0 %1 %2 %3 %4 %5 %6 %7 %8 %9
for %%q in (1 2 3 4 5 6 7 8 9 10) do shift 
set args=%args% %0 %1 %2 %3 %4 %5 %6 %7 %8 %9

python src/_genSGAN2.py --model models/%model% --out_dir _out/%name% --size %size% --frames %frames%  %args%
goto ff

:test
python src/_genSGAN2.py --model models/%name%.pkl --out_dir _out/%name% --frames 200-20 ^
%3 %4 %5 %6 %7 %8 %9

:ff
ffmpeg -y -v warning -i _out\%name%\%%06d.jpg -c:v mjpeg -q:v 2 _out/%name%-%2.avi
rem rmdir /s /q _out\%name%

goto end


:help
echo Usage: gen model x-y framecount-transit
echo e.g.:  gen ffhq-1024 1280-720 100-25

:end
