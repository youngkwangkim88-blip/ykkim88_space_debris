# A1 파편 광학·하늘 배경 재현 예제

Python 3.10 환경에서 아래 두 명령을 실행한다. 의존성을 설치한 뒤에는 기본 예제에 필요한 관측 영상과 모형 입력이 모두 이 폴더에 있다. [영문 실행 안내](README.md) · [원자료와 이용 조건](SOURCES.md)

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

- **파편** — `python fragment.py`는 10 cm 회색판의 초기 자세·회전축 100쌍을 적산한다. 4초 E12에서 조명을 같은 8월 조건으로 유지하면 약한 하늘과 강한 하늘의 SNR 10 거리 중앙값은 각각 306.0 km와 275.5 km다. `python fragment.py --shape cuboid --material solar_cell --band E5 --exposure 0.5`로 형상·재질·대역·노출을 바꿀 수 있다.
- **하늘** — `python sky.py`는 황도광, 등방 잔여광, 확산 은하광, 미분해 별빛과 달 미광을 성분별로 계산한다. 7월 관측 기하에서 달을 28.5°에 놓은 사례는 별도의 가상 조건이다.
- **단노출 합성** — `python coadd.py`는 같은 네 초 동안 수집한 광자를 단일 4초 노출 또는 2·0.5·0.1초 판독으로 나누어, 한 장으로 합쳤을 때와 시간순 원본을 보존했을 때의 가중 SNR을 대조한다.
- **TESS 실측** — `python tess.py`는 동봉한 25×25화소 FITS에서 731개 시점으로 배경 78.947314 e⁻/화소/s를 얻는다. 총배경 오차 9.80885%는 별도 관측창 250개의 측광표를 방문·표본 순으로 집계한 값이다.
- **통계 검산** — `python factors.py`는 저장된 SNR 격자에서 표 5.1을 계산한다. `python reproduce.py`는 위 예제와 광자·판독잡음을 새로 뽑는 512회 분산 검산을 함께 실행한다.
