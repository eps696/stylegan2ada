@echo off
chcp 65001 > NUL

python src/model_convert.py --source %1 %2 %3 %4 %5 %6 %7 %8 %9

