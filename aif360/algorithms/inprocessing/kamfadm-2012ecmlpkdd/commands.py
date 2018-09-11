# minimal polyfill for commands in python3

import subprocess

def getoutput(cmd):
    output = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout
    output = str(output, 'utf-8')
    return output

