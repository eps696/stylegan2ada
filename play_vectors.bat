@echo off
if "%1"=="" goto help

python src/_play_vectors.py --model models/%1 --base_lat _in/%2 --vector_dir _in/%3 --out_dir _out/%~n1-%~n3 ^
%4 %5 %6 %7 %8 %9

ffmpeg -y -v warning -i _out\%~n1-%~n3\%%06d.jpg -c:v mjpeg -q:v 2 %~n1-%~n3.avi
rmdir /s /q _out\%~n1-%~n3

goto end 

:help
echo Usage: play_vectors model latentsdir vector
echo  e.g.: play_vectors ffhq-1024-f npy age.npy

:end