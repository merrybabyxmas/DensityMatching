[가상환경 생성]

conda create -y -n rlenv python=3.10
conda activate rlenv
# export PYTHONNOUSERSITE=1 (제가 실행할땐 이 명령어를 적어야 했습니다)
pip install -r requirements.txt

[실행]

python run_policy.py


