"""Optional Locust scenario:  locust -f loadtest/locustfile.py --host http://127.0.0.1:8000"""
import random

from locust import HttpUser, between, task

PROMPTS = ["What is 17 * 23?", "Write a haiku about autumn.", "Explain recursion simply.",
           "Prove that sqrt(2) is irrational.", "Summarize the plot of Hamlet in two sentences."]


class RouterUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def route_only(self):
        self.client.post("/v1/route", json={"messages": [{"role": "user", "content": random.choice(PROMPTS)}]})

    @task(1)
    def full(self):
        self.client.post("/v1/chat/completions",
                         json={"messages": [{"role": "user", "content": random.choice(PROMPTS)}], "max_tokens": 64})
