import time
import os
current_milli_time = lambda: int(round(time.time() * 1000))

start = 0
finish = 0
span = 0
task = ""
save_dir = "./time_measure"

def start_measure(task_name):
    global task
    task = task_name
    global start
    start = current_milli_time()
    return

def finish_measure():
    global finish
    global span
    finish = current_milli_time()
    span = finish - start
    result_string = task + ": " + str(span)

    file_name = task
    save_file = save_dir + file_name

    file = open(save_file, 'a')
    file.write(result_string)
    file.close()

    return result_string

