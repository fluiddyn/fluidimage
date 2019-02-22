
import os
import subprocess

from path_images import path_base


def call_bash(commands):
    subprocess.call(commands, shell=True)

keys = ['2001A', '2003C', '2005C']

str_glob = {
    2001: 'http://www.pivchallenge.org/pub/{l}/{l}.zip',
    2003: 'http://www.pivchallenge.org/pub03/{l}all.zip',
    2005: 'http://www.pivchallenge.org/pub05/{l}/{l}.zip'
}


for key in keys:
    if key in ['2005A']:
        raise NotImplementedError
    year = int(key[:4])
    l = key[4]

    path = os.path.join(path_base, f'PIV{year}' + l)
    path_zip = os.path.join(path, 'data.zip')
    path_images = os.path.join(path, 'Images')

    if not os.path.exists(path_images):
        os.makedirs(path_images)

    if len(os.listdir(path_images)) == 0:
        if not os.path.exists(path_zip):
            # download the zip file
            command = 'wget ' + str_glob[year].format(l=l) + ' -O ' + path_zip
            print(command)
            call_bash(command)

        command = 'unzip ' + path_zip + ' -d ' + path_images
        print(command)
        call_bash(command)

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
