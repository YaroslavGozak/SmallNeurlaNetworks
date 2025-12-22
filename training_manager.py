
import argparse
from dataclasses import dataclass
import multiprocessing
import queue
import time
import tkinter as tk
from tkinter import ttk
from enum import Enum
from typing import cast

import psutil

class NotificationType(Enum):
    COMPLETE = 1
    NEW_NETWORK = 2
    NETWORK_UPDATE = 3
    TRAINING_TOTAL_PROGRESS = 4

class NetworkUpdateData:
    config: str
    epoch: int
    total_epoch: int
    img_processed: int
    total_imgs: int
    accuracy_list: list[float]

class ProgressUpdateData:
    current: int
    total: int

@dataclass
class QueueItem:
    type: NotificationType
    data: object

class TrainingManager:
    is_training_suspended = False
    pause_button: ttk.Button = None
    managed_pid = None

    def update_button_ui(self):
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
        self.update_button_ui()
        print('Training suspended')

    def resume_process(self):
        try:
            os_process = psutil.Process(self.managed_pid)
            os_process.resume()
        except psutil.NoSuchProcess:
            print('Process already destroyed')
        self.is_training_suspended = False
        self.update_button_ui()
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
        return ttk.Button(self.frm, text="Pause", command=self.suspend_process).grid(column=3, row=9)
    
    def create_resume_button(self):
        return ttk.Button(self.frm, text="Resume", command=self.resume_process).grid(column=3, row=9)
    
    def destroy_button(self, button: ttk.Button):
        if button is not None:
            button.destroy()

    def process_notification(self, notification: QueueItem):
        if notification.type == NotificationType.COMPLETE:
            print("Complete signal received. Exiting...")
            self.exit_program()

        if notification.type == NotificationType.NETWORK_UPDATE:
            data = cast(NetworkUpdateData, notification.data)
            try:
                epoch_percent = data.epoch / data.total_epoch * 100
                img_percent = data.img_processed / data.total_imgs * 100
                if epoch_percent < 0:
                    print(f'Invalid progress percent for epoch. Value {epoch_percent}')
                    return
                self.epoch_progress_bar['value'] = epoch_percent
                self.epoch_progress_label.config(text=f' {data.epoch} / {data.total_epoch}')
                if img_percent < 0:
                    print(f'Invalid progress percent for image. Value {img_percent}')
                    return
                self.img_progress_bar['value'] = img_percent
                self.img_progress_label.config(text=f' {data.img_processed} / {data.total_imgs}')
                self.config_label.config(text=data.config)
                if data.accuracy_list and len(data.accuracy_list) > 0:
                    joined_accuracies = ', '.join([str(round(acc*100,2)) + '%' for acc in data.accuracy_list[-3:]])
                    self.epoch_acc_label.config(text=f'Last accuracy: {joined_accuracies}')
                last_update = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                self.root.title(f'CNN Trainer Manager - Last update {last_update}')
            except Exception as e:
                print(f'Could not calculate update: {e}')
            

        if notification.type == NotificationType.NEW_NETWORK:
            name = str(notification.data)
            self.config_label.config(text=name)
            self.epoch_progress_bar['value'] = 0
            self.img_progress_bar['value'] = 0

        if notification.type == NotificationType.TRAINING_TOTAL_PROGRESS:
            data = cast(ProgressUpdateData, notification.data)
            percent = data.current / data.total * 100
            self.overall_progress_bar['value'] = percent
            self.overall_progress_label.config(text=f' {data.current} / {data.total}')

    def check_for_notification(self, notification_queue: multiprocessing.Queue):
        notification: QueueItem = None
        try:
            notification = notification_queue.get_nowait()
        except queue.Empty:
            pass

        if notification is not None:
            self.process_notification(notification)

        self.root.after(100, lambda: self.check_for_notification(notification_queue))
        
    def create_management_form(self, queue: multiprocessing.Queue):
        self.root = tk.Tk()
        self.root.geometry("900x300")
        self.frm = ttk.Frame(self.root, padding=10)
        self.frm.grid()
        ttk.Label(self.frm, text="CNN trainer manager").grid(column=0, row=0)

        ttk.Label(self.frm, text="Overall Progress:").grid(column=0, row=2)
        self.overall_progress_bar = ttk.Progressbar(self.frm, orient="horizontal", length=300, mode="determinate", maximum=100, value=0)
        self.overall_progress_bar.grid(column=1, row=2, columnspan=2)
        self.overall_progress_label = ttk.Label(self.frm, text="")
        self.overall_progress_label.grid(column=3, row=2)

        ttk.Label(self.frm, text="Epoch Progress:").grid(column=0, row=4)
        self.epoch_progress_bar = ttk.Progressbar(self.frm, orient="horizontal", length=300, mode="determinate", maximum=100, value=0)
        self.epoch_progress_bar.grid(column=1, row=4, columnspan=2)
        self.epoch_progress_label = ttk.Label(self.frm, text="")
        self.epoch_progress_label.grid(column=3, row=4)
        self.epoch_acc_label = ttk.Label(self.frm, text="Last accuracy: N/A")
        self.epoch_acc_label.grid(column=4, row=4)

        ttk.Label(self.frm, text="Images Progress:").grid(column=0, row=6)
        self.img_progress_bar = ttk.Progressbar(self.frm, orient="horizontal", length=300, mode="determinate", maximum=100, value=0)
        self.img_progress_bar.grid(column=1, row=6, columnspan=2)
        self.img_progress_label = ttk.Label(self.frm, text="")
        self.img_progress_label.grid(column=3, row=6)

        self.config_label = ttk.Label(self.frm, text="")
        self.config_label.grid(column=0, row=8, columnspan=4)

        ttk.Button(self.frm, text="Stop", command=self.exit_program).grid(column=0, row=9)
        self.pause_button = self.create_pause_button()
        self.resume_button = None

        self.root.after(100, lambda: self.check_for_notification(queue))
        self.root.mainloop()

    def start_managed_process(self, delegate, args=()):
        delegate_to_master_queue: "queue.Queue[QueueItem]" = multiprocessing.Queue()
        print(f'Created queue: {delegate_to_master_queue}')

        process = multiprocessing.Process(target=self.run_with_notify_completion, args=(delegate, delegate_to_master_queue, args))
        process.start()
        self.managed_pid = process.pid
        print(f'Process id after start: {self.managed_pid}')
        self.create_management_form(delegate_to_master_queue)
        process.join()
        print('Processes joined')
        print('Exiting...')

    def run_with_notify_completion(self, delegate, notification_queue: multiprocessing.Queue, args=()):
        try:
            delegate(notification_queue, *args)
        except Exception as e:
            import traceback
            print('Error occured during running underlying delegate method. Aborting......')
            print(e)
            print(traceback.format_exc())
        print('Delegate completed. Sending exit notification...')
        notification_queue.put_nowait(QueueItem(NotificationType.COMPLETE, data={}))


def run_training(delegate_to_master_queue: multiprocessing.Queue, args: argparse.Namespace):
    i = 0
    while True:
        i+=1
        print(f'{args.mode} {i}')
        time.sleep(0.5)
        try:
            msg = QueueItem(NotificationType.NETWORK_UPDATE, data=(100/20*i))
            print(f'Putting message to queue {delegate_to_master_queue}. Message: {msg}')
            delegate_to_master_queue.put_nowait(msg)
        except queue.Full:
            print('Queue full')
            pass

        if i == 20:
            msg = QueueItem(NotificationType.COMPLETE, data={})
            delegate_to_master_queue.put_nowait(msg)
            time.sleep(5)
            return

if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(
                            prog='program',
                            description='Trains neural networks on MNIST or Cifar datasets',
                            epilog='Text at the bottom of help')

    parser.add_argument('mode', choices=['train', 'test', 'performance'])
    args = parser.parse_args()

    manager = TrainingManager()
    manager.start_managed_process(run_training, args=(args,))