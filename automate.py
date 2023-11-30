"""
Code version to query API locally or remotely
"""
import argparse
import json
import os
import time
import tqdm
import logging
import threading
from queue import Queue
from typing import Optional, Dict, Tuple
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/") + "/inference"
API_AUTH = os.getenv("API_AUTH", "master:password")
# Add your API URLs here or maybe dotenv
API_URLS = {
    API_URL,
}
API_AUTHS = [API_AUTH]

JOB_LOGS_DATABASE = {}

DEFAULT_INFERENCE_KWARGS = {
    "temperature": 0.2,
    "top_p": 0.95,
    "max_new_tokens": 256,
    "use_cache": True,
    "do_sample": True,
}


# This is the data format for the API
#    {
#        "num_samples": 5,
#        "inference_kwargs": {
#            "temperature": 0.2,
#            "top_p": 0.95,
#            "max_new_tokens": 256,
#            "use_cache": true,
#            "do_sample": true,
#        },
#        "delimeter": ". ",
#        "text_or_path": "1girl, white hair, short hair, lightblue eyes, flowers, light, sitting",
#        "image_or_path": "https://github.com/AUTOMATIC1111/stable-diffusion-webui/assets/35677394/f6929d4d-5991-4c10-b013-0743ffc8e207",
#        "prompt_format": "",
#    }
def sanitize_data(data: dict) -> dict:
    """
    Sanitize data
    """
    sanitized_data = {
        "num_samples": data.get("num_samples", 5),
        "inference_kwargs": data.get(
            "inference_kwargs", DEFAULT_INFERENCE_KWARGS.copy()
        ),
        "delimeter": data.get("delimeter", ". "),
        "text_or_path": data["text_or_path"],
        "image_or_path": data["image_or_path"],
        "prompt_format": data.get("prompt_format", ""),
    }
    return sanitized_data


class JobStatus:
    """
    Job status
    """

    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


class QueryHandler:
    """
    Handles queries to API
    """

    def __init__(self, api_url: str, api_auth: str, _result_file: str = None):
        # if ends with / remove it
        self.api_url = api_url.rstrip("/") + "/inference"
        self.api_auth = api_auth
        self.session = requests.Session()
        self.session.auth = self.api_auth.split(":")
        self.queue = Queue()
        self.thread = threading.Thread(target=self._process_queue)
        self.job_count = 0
        self.job_done = 0
        self.iterator = None
        self.iterator_thread = None
        self.job_logs_database = {}
        self.result_file = _result_file

    def get_progress(self) -> Tuple[int, int]:
        """
        Get progress
        """
        return self.job_done, self.job_count

    def _write_result(self, job_id: int, result: dict):
        """
        Write result to database as jsonl
        """
        if self.result_file is None:
            return
        with open(self.result_file, "a", encoding="utf-8") as f:
            json.dump({"job_id": job_id, "result": result}, f)
            f.write("\n")

    def _query(self, data: dict, job_id: int) -> Optional[dict]:
        """
        Query API with refined exception handling
        """
        try:
            if self.job_logs_database.get(job_id) != JobStatus.NOT_STARTED:
                return None
            data = sanitize_data(data)
            self.job_logs_database[job_id] = JobStatus.RUNNING
            response = self.session.post(self.api_url, json=data)
            if response.status_code != 200:
                self.job_logs_database[job_id] = JobStatus.FAILED
                return None
            self.job_logs_database[job_id] = JobStatus.FINISHED
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Request exception for job {job_id}: {e}")
            self.job_logs_database[job_id] = JobStatus.FAILED
            return None
        except Exception as e:
            logging.error(f"Unexpected exception for job {job_id}: {e}")
            self.job_logs_database[job_id] = JobStatus.FAILED
            return None
        finally:
            self.job_done += 1

    def _process_queue(self):
        """
        Process queue
        """
        while True:
            if self.queue.empty():
                time.sleep(0.1)
                continue
            data, job_id = self.queue.get()
            result = self._query(data, job_id)
            if result is not None:
                self._write_result(job_id, result)
            self.queue.task_done()

    def start(self):
        """
        Start queue
        """
        self.thread.start()

    def wait(self):
        """
        Finalize the jobs.
        """
        if self.iterator_thread is not None:
            self.iterator_thread.join()
        self.queue.join()

    def append_job(self, job_id: int, data: dict) -> int:
        """
        Append job
        """
        self.job_count += 1
        self.queue.put(data, job_id)
        return job_id

    def _from_iterator(self):
        while True:
            if self.iterator is None or len(self.queue) > 0:
                time.sleep(0.1)
            try:
                self.append_job(*next(self.iterator))
            except StopIteration:
                self.iterator = None
                # kill thread
                self.iterator_thread = None
                return None

    def register_iterator(self, iterator):
        """
        Registers iterator that yields (data, job_id)
        """
        self.iterator = iterator
        self.iterator_thread = threading.Thread(target=self._from_iterator)
        self.iterator_thread.start()

    def get_job_logs_database(self):
        """
        Returns job logs database
        """
        return self.job_logs_database


def main(urls, auths, job_database: Dict[int, Dict]):
    """
    Main function
    """
    job_handlers: Dict[str, QueryHandler] = {}
    for _i, url in enumerate(urls):
        timestamp = int(time.time())
        handler = QueryHandler(url, auths[_i], f"results_{timestamp}.jsonl")
        job_handlers[url] = handler
        handler.start()

    for handlers in job_handlers.values():
        handlers.register_iterator(job_database.items())

    pbar = tqdm.tqdm(total=len(job_database))
    try:
        while any(
            handler.job_done < handler.job_count for handler in job_handlers.values()
        ):
            pbar.update(
                sum(handler.job_done for handler in job_handlers.values()) - pbar.n
            )
            time.sleep(0.1)
    finally:
        pbar.close()

    for handler in job_handlers.values():
        handler.wait()
    updated_job_logs_database = {}
    for handler in job_handlers.values():
        updated_job_logs_database.update(handler.get_job_logs_database())
    # update job logs database
    JOB_LOGS_DATABASE.update(updated_job_logs_database)
    # join job_results
    job_results = {}
    for handler in job_handlers.values():
        result_file = handler.result_file
        if result_file is None:
            continue
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                job_results[result["job_id"]] = result["result"]
    print("All jobs finished")
    return job_results


def test_job():
    """
    Test job
    """
    # Updates JOB_LOGS_DATABASE with a test job
    job_database = {}
    for i in range(10):
        JOB_LOGS_DATABASE[i] = JobStatus.NOT_STARTED
        job_database[i] = {
            "num_samples": 5,
            "inference_kwargs": {
                "temperature": 0.2,
                "top_p": 0.95,
                "max_new_tokens": 256,
                "use_cache": True,
                "do_sample": True,
            },
            "delimeter": ". ",
            "text_or_path": "1girl, white hair, short hair, lightblue eyes, flowers, light, sitting",
            "image_or_path": "https://github.com/AUTOMATIC1111/stable-diffusion-webui/assets/35677394/f6929d4d-5991-4c10-b013-0743ffc8e207"
        }
    test_job_results = main(API_URLS, API_AUTHS, job_database)
    return test_job_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="+", default=["localhost:8000"])
    parser.add_argument("--auths", nargs="+", default=["master:password"])
    parser.add_argument("--job-logs-database", default="job_logs_database.json") # contains job_id, status
    parser.add_argument("--job-database", default="job_database.json") # contains job_id, data
    parser.add_argument("--test-job", action="store_true")
    args = parser.parse_args()
    if os.path.exists(args.job_logs_database):
        with open(args.job_logs_database, "r", encoding="utf-8") as f:
            JOB_LOGS_DATABASE = json.load(f)
    if os.path.exists(args.job_database):
        with open(args.job_database, "r", encoding="utf-8") as f:
            jobs_database = json.load(f)
    if args.test_job:
        test_result = test_job()
        assert len(test_result) == 10, "Failed to finish test job"
        assert all(v == JobStatus.FINISHED for v in JOB_LOGS_DATABASE.values()), "Failed to finish test job"
        exit(0)
    job_results_merged = main(
        args.urls, args.auths, jobs_database
    )  # Note, job_database should now be locked
    with open(args.job_logs_database, "w", encoding="utf-8") as f:
        json.dump(JOB_LOGS_DATABASE, f, indent=4)
    with open("job_results.json", "w", encoding="utf-8") as f:
        json.dump(job_results_merged, f, indent=4)
