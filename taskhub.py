#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 09:53:33 2021

@author: ljia
"""

if __name__ == '__main__':
	tasks = [
		{'path': '.',
         'file': 'run_job_treelet.py'
		 },
		]

	import os
	for t in tasks:
		print(t['file'])
		command = ''
		command += 'cd ' + t['path'] + '\n'
		command += 'python3 ' + t['file'] + '\n'
# 		command += 'cd ' + '/'.join(['..'] * len(t['path'].split('/'))) + '\n'
		os.system(command)
