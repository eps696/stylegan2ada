@echo off

: multicrop images to squares of specific size 
: %1 = video file or directory with images
: %2 = size
: %3 = shift step 

if exist %1\* goto dir

:video
echo .. cropping video
if not exist "%~dp1\%~n1-tmp" md "%~dp1\%~n1-tmp"
ffmpeg -y -v error -i %1 -q:v 2 "%~dp1\%~n1-tmp\%~n1-c-%%06d.png"
python ../src/util/multicrop.py --in_dir %~dp1/%~n1-tmp --out_dir %~dp1/%~n1-sub --size %2 --step %3
rmdir /s /q %~dp1\%~n1-tmp
goto end

:dir
echo .. cropping images
python ../src/util/multicrop.py --in_dir %1 --out_dir %~dp1/%~n1-sub --size %2 --step %3
goto end

:end