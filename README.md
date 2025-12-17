<<<<<<< HEAD
python -m venv venv

# Windows

venv\Scripts\activate

# macOS/Linux

source venv/bin/activate

pip install -r requirements.txt
python main_ppo.py --episodes 1
=======
pytorch를 이용해서 동방 게임(홍마향)을 AI가 자동으로 플레이하는 프로그램을 만드는 중

목표는 루나틱 난이도 클리어

현재는 저해상도 150x150 즈음의 게임 화면으로 AI에게 학습시키고 있으며, 좀 더 익숙해지면 차차 해상도를 늘려나가며 AI의 정확도를 늘릴 예정.

사용법 :
- 동방 게임(홍마향)을 실행. 창모드 필수.
- practice모드 -> 캐릭,무기 고르고 게임화면 진입.
- 이후, 터미널을 통해 main.py(또는 main_ppo.py)를 실행 시킨다. 명령어는 python main.py
- 그러면 ai가 '동방홍마향' 글자가 들어간 프로그램을 감지->해당 프로그램으로 포커스 옯기고 작동.
- 기본목숨값은 3으로 설정됨.
- -아직 미완성.

<img width="639" height="504" alt="스크린샷 2025-12-15 194333" src="https://github.com/user-attachments/assets/d9c8766c-71e9-4e81-9bed-83b632aa3a5e" />
>>>>>>> 3052d9bbf8de3e57f0345ac38f895224f9e04a30
