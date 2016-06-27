
import os
import subprocess


def call_bash(commands):
    subprocess.call(commands, shell=True)

for l in ['A', 'B', 'C', 'E']:
    path = 'PIV2001' + l
    if not os.path.exists(path):
        commands = """
wget http://www.pivchallenge.org/pub/{l}/{l}.zip
unzip {l}.zip -d PIV2001{l}
""".format(l=l)
        print(commands)
        call_bash(commands)

"""
2003A
http://www.pivchallenge.org/pub03/Aall.zip

2003B
http://www.pivchallenge.org/pub03/Ball.zip

2003C
http://www.pivchallenge.org/pub03/Call.zip

2005A
http://www.pivchallenge.org/pub05/A/A1.zip
...
http://www.pivchallenge.org/pub05/A/A4.zip

2005B
http://www.pivchallenge.org/pub05/B/B.zip

2005C
http://www.pivchallenge.org/pub05/C/C.zip
"""
