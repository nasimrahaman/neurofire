import sys
import time
import logging
from queue import Empty

from torch.utils.data.dataloader import ExceptionWrapper
import torch.multiprocessing as mp


def _worker_loop(dataset,
                 job_queue: mp.Queue, result_queue: mp.Queue,
                 interrupt_event: mp.Event):
    logger = logging.getLogger("worker_loop")
    logger.debug("Worker started.")
    while True:
        logger.debug("Trying to fetch from job_queue.")
        if interrupt_event.is_set():
            logger.debug("Received interrupt signal, breaking.")
            break
        try:
            # This assumes that the job_queue is fully populated before the worker is started.
            index = job_queue.get_nowait()
            logger.debug("Fetch successful.")
        except Empty:
            logger.debug("Queue empty, setting up poison pill.")
            index = None
        if index is None or interrupt_event.is_set():
            logger.debug("Fetched poison pill or received interrupt signal, breaking.")
            break
        try:
            logger.debug("Sampling index {} from dataset.".format(index))
            sample = dataset[index]
        except Exception:
            logger.debug("Dataset threw an exception.".format(index), exc_info=1)
            result_queue.put((index, ExceptionWrapper(sys.exc_info())))
        else:
            logger.debug("Putting sample at index {} in the result queue.".format(index))
            result_queue.put((index, sample))


class SimpleParallelLoader(object):
    def __init__(self, dataset, num_workers):
        # Publics
        self.dataset = dataset
        self.num_workers = num_workers
        self.job_queue = mp.Queue()
        self.result_queue = mp.Queue()
        # Privates
        self._processes = []
        self._workers_started = False
        self._workers_killed = False
        self._interrupt_event = mp.Event()
        self._samples_outstanding = len(dataset)

    def start_workers(self):
        logger = logging.getLogger('SimpleParallelLoader.start_workers')
        # Give'em jobs
        logger.debug("Putting {} jobs in job_queue.".format(len(self.dataset)))
        for index in range(len(self.dataset)):
            self.job_queue.put(index)
        # Start workers as daemons
        for process_num in range(self.num_workers):
            logger.debug("Starting worker {} of {}.".format(process_num, self.num_workers))
            worker = mp.Process(target=_worker_loop,
                                args=(self.dataset,
                                      self.job_queue, self.result_queue,
                                      self._interrupt_event))
            worker.daemon = True
            worker.start()
            self._processes.append(worker)
        self._workers_started = True

    def kill_workers(self):
        logger = logging.getLogger('SimpleParallelLoader.kill_workers')
        logger.debug("Asking all workers to die.")
        self._interrupt_event.set()
        while True:
            workers_dead = all([not worker.is_alive() for worker in self._processes])
            if not workers_dead:
                logger.debug("Waiting for workers to die.")
                time.sleep(1)
                continue
            else:
                logger.debug("Workers dead.")
                break
        self._workers_killed = True

    def clean_up(self):
        logger = logging.getLogger('SimpleParallelLoader.clean_up')
        if not self._workers_killed:
            logger.debug("Killing workers.")
            self.kill_workers()
        else:
            logger.debug("Workers are already dead.")

    def next_batch(self, batch_size=1):
        logger = logging.getLogger('SimpleParallelLoader.next_batch')
        if not self._workers_started:
            # This could happen the first time next_batch is called
            logger.debug("Starting workers.")
            self.start_workers()
        if self._samples_outstanding < 1:
            logger.debug("No batches outstanding, cleaning up and returning an empty list.")
            self.clean_up()
            return []
        # Fetch batches
        batch = []
        exhausted = False
        for sample_num in range(batch_size):
            if self._samples_outstanding > 0:
                logger.debug("Trying to fetch sample {} of {}.".format(sample_num, batch_size))
                result = self.result_queue.get()
                if isinstance(result[-1], ExceptionWrapper):
                    # Kill workers and die
                    logger.debug("Received exception, cleaning up.")
                    self.clean_up()
                    logger.error("Raising exception.")
                    raise result[-1].exc_type(result[-1].exc_msg)
                else:
                    batch.append(result)
                self._samples_outstanding -= 1
                logger.debug("Fetch successful. There are still {} samples to go."
                             .format(self._samples_outstanding))
            else:
                logger.debug("Dataset exhausted.")
                exhausted = True
        if exhausted:
            logger.debug("Cleaning up.")
            self.clean_up()
        logger.debug("Returning batch of len {}.".format(len(batch)))
        return batch
