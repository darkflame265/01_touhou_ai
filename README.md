#사용법.
1. 가상환경 진입
venv\Scripts\activate

2. 부족한 라이브러리 설치.
pip install -r requirements.txt

3. main_ppo.py 파일 실행.
python main_ppo.py --episodes 1

=======
pytorch를 이용해서 동방 게임(홍마향)을 AI가 자동으로 플레이하는 프로그램을 만드는 중

목표는 루나틱 난이도 클리어

사용법 :
- 동방 게임(홍마향)을 실행. 창모드 필수.
- practice모드 -> 캐릭,무기 고르고 게임화면 진입.
- 이후, 터미널을 통해 main.py(또는 main_ppo.py)를 실행 시킨다. 명령어는 python main.py
- 그러면 ai가 '동방홍마향' 글자가 들어간 프로그램을 감지->해당 프로그램으로 포커스 옯기고 작동.
- 기본 목숨값은 3으로 가정.


<img width="639" height="504" alt="스크린샷 2025-12-15 194333" src="https://github.com/user-attachments/assets/d9c8766c-71e9-4e81-9bed-83b632aa3a5e" />

<img width="634" height="507" alt="스크린샷 2026-01-05 124444" src="https://github.com/user-attachments/assets/60490f1e-6c81-4773-a674-b8bad4e22b0c" />

