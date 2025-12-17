python -m venv venv

# Windows

venv\Scripts\activate

# macOS/Linux

source venv/bin/activate

pip install -r requirements.txt
python main_ppo.py --episodes 1
