"""Bounded TTL state; Redis failures are visible, not silently ignored."""
import json
import time
from collections import OrderedDict


class StateStore:
    def __init__(self, redis_url=None, maximum=500):
        self.maximum = maximum
        self.local = OrderedDict()
        self.redis = None
        if redis_url:
            from redis import Redis
            self.redis = Redis.from_url(redis_url, decode_responses=True, socket_timeout=3)
            self.redis.ping()

    def get(self, key):
        if self.redis:
            data = self.redis.get("eki:" + key)
            return json.loads(data) if data else None
        item = self.local.get(key)
        if item and item[0] > time.monotonic():
            return json.loads(item[1])
        self.local.pop(key, None)
        return None

    def put(self, key, value, ttl):
        data = json.dumps(value)
        if self.redis:
            self.redis.setex("eki:" + key, ttl, data)
        else:
            self.local[key] = (time.monotonic() + ttl, data)
            self.local.move_to_end(key)
            while len(self.local) > self.maximum:
                self.local.popitem(last=False)

    def close(self):
        if self.redis:
            self.redis.close()
