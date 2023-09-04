# run_script.py
import subprocess
import os

QLWeightsFile = 'QLWeightsMyTeam_update.txt'
sub_result_of = []
for i  in range(100):
    command = ["python", "capture.py",'-r','myTeam.py','-b','berkeleyTeam.py','-n','10','-Q']
    subprocess.run(command)
    if os.path.exists(QLWeightsFile):
        with open(QLWeightsFile, "r") as file:
            QLWeights = eval(file.read())
            sub_result_of.append((QLWeights))


