@echo off
chcp 65001 > NUL

python src/projector.py --model=models/%1 --in_dir=_in/%2 --out_dir=_out/proj/%2 ^
--save_video ^
%3 %4 %5 %6 %7 %8 %9

