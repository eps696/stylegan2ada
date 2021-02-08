@echo off
if "%1"=="" goto help

python src/_play_dlatents.py --model models/%1 --dlatents _in/%2 --out_dir _out/%~n1-%~n2 --fstep %3 --size %4 ^
%5 %6 %7 %8 %9

ffmpeg -y -v warning -i _out\%~n1-%~n2\%%06d.jpg -c:v mjpeg -q:v 2 _out/%~n1-%~n2.avi
rem rmdir /s /q _out\%~n1-%~n2

goto end 

:help
echo Usage: play_dlatents model latentsdir fstep size
echo  e.g.: play_dlatents ffhq-1024-f npy 25 1920-1080

:end