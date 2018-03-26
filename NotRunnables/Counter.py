import time
current_milli_time = lambda: int(round(time.time() * 1000))

class Counter():
    start = 0
    finish = 0
    span = 0
    task = ""
    file_name = "./time_measure"
    unit = "ms"

    def start_measure(self, task_name):
        self.task = task_name
        self.start = current_milli_time()
        return

    def finish_measure(self):
        self.finish = current_milli_time()
        self.span = self.finish - self.start
        result_string = self.task + ": " + str(self.span) + " " + self.unit

        with open(self.file_name, 'a') as file:
            file.write(result_string)
        return result_string

