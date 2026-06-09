# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import pickle

from job_submission import Job

if __name__ == "__main__":
    path = "/root/job.pkl"
    with open(path, "rb") as f:
        job: Job = pickle.loads(f.read())

    job.run()
