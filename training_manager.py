
import argparse
import multiprocessing
import os
import time
from tkinter import Tk
from tkinter import ttk

import psutil

class TrainingManager:
    is_training_suspended = False
    pause_button: ttk.Button = None
    managed_pid = None

    def update_button(self):
        if self.is_training_suspended:
            self.destroy_button(self.pause_button)
            self.resume_button = self.create_resume_button()
        else:
            self.destroy_button(self.resume_button)
            self.resume_button = self.create_pause_button()

    def suspend_process(self):
        try:
            os_process = psutil.Process(self.managed_pid)
            os_process.suspend()
        except psutil.NoSuchProcess:
            print('Process already destroyed')
        self.is_training_suspended = True
        self.update_button()
        print('Training suspended')

    def resume_process(self):
        try:
            os_process = psutil.Process(self.managed_pid)
            os_process.resume()
        except psutil.NoSuchProcess:
            print('Process already destroyed')
        self.is_training_suspended = False
        self.update_button()
        print('Training resumed')

    def exit_program(self):
        try:
            os_process = psutil.Process(self.managed_pid)
            os_process.kill()
        except psutil.NoSuchProcess:
            print('Process already destroyed')
        self.root.destroy()
        print('Training stopped')

    def create_pause_button(self):
        return ttk.Button(self.frm, text="Pause", command=self.suspend_process).grid(column=1, row=1)
    
    def create_resume_button(self):
        return ttk.Button(self.frm, text="Resume", command=self.resume_process).grid(column=1, row=1)
    
    def destroy_button(self, button: ttk.Button):
        if button is not None:
            button.destroy()

    def check_for_process_completion(self, parent_conn):
        if parent_conn.poll():
            print("Signal received. Exiting...")
            self.exit_program()
        self.root.after(100, lambda: self.check_for_process_completion(parent_conn))
        
    def create_management_form(self, parent_conn):
        self.root = Tk()
        self.root.geometry("300x300")
        self.frm = ttk.Frame(self.root, padding=10)
        self.frm.grid()
        ttk.Label(self.frm, text="CNN trainer manager").grid(column=0, row=0)
        ttk.Button(self.frm, text="Stop", command=self.exit_program).grid(column=0, row=1)
        self.pause_button = self.create_pause_button()
        self.resume_button = None
        self.root.after(100, lambda: self.check_for_process_completion(parent_conn))
        self.root.mainloop()

    def start_managed_process(self, delegate, args=()):
        parent_conn, child_conn = multiprocessing.Pipe()
        process = multiprocessing.Process(target=self.run_with_notify_completion, args=(delegate, child_conn, args))
        self.managed_pid = process.pid
        process.start()
        self.managed_pid = process.pid
        print(f'Process id after start: {self.managed_pid}')
        self.create_management_form(parent_conn)
        process.join()
        print('Processes joined')
        print('Exiting...')

    def run_with_notify_completion(self, delegate, child_conn, args=()):
        try:
            delegate(*args)
        except Exception as e:
            print('Error occured during running underlying delegate method. Aborting......')
            print(e)
        print('Delegate completed. Sending exit notification...')
        child_conn.send("Completed")


def run_training(args: argparse.Namespace):
    i = 0
    while True:
        i+=1
        print(f'{args.mode} {i}')
        time.sleep(0.1)
        if i == 20:
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                            prog='program',
                            description='Trains neural networks on MNIST or Cifar datasets',
                            epilog='Text at the bottom of help')

    parser.add_argument('mode', choices=['train', 'test', 'performance'])
    args = parser.parse_args()

    manager = TrainingManager()
    manager.start_managed_process(run_training, args=(args,))