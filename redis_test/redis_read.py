
import redis
import msgpack

class RedisBase():
    def __init__(self, host, port=6379, db=9, socket_connect_timeout=1):
        self.__r = redis.StrictRedis(host=host, port=port, db=db, socket_connect_timeout=socket_connect_timeout)

    def write(self, key, value):
        try:
            self.__r.set(key.encode('utf-8'), msgpack.dumps(value))
        except Exception as e:
            print(Exception, e)
            return None

    def read(self, key):
        try:
            return msgpack.unpackb(self.__r.get(key.encode('utf-8')))
        except Exception as e:
            print(Exception, e)
            print("data is not existed")
            return None
        
    def exists(self, key):
        try:
            res = self.__r.exists(key)
            return res == 1
        except Exception as e:
            print("==== error in redis : ", e)
            return False
    def key_type_(self, key):
        return self.__r.type(key).decode("utf-8")
    def key_len_(self, key, type):
        if type == 'string':
            return self.__r.strlen(key)
        elif type == 'hash':
            return self.__r.hlen(key)
    
    def key_ttl_(self, key):
        ttl = self.__r.ttl(key)
        if ttl > 0:
            return f"{ttl}s"
        elif ttl == -1:
            return "No limit"
        else:
            return ""
    def keys(self):
        try:
            ks = self.__r.keys()
            return [[self.key_type_(k), k.decode("utf-8"), self.key_ttl_(k), self.key_len_(k, self.key_type_(k))] for k in ks]
        except Exception as e:
            return self.redis_error(e, [])
        
    def redis_error(self, e, res=None):
        print(Exception, e)
        return res
    

if __name__ == "__main__":
    import time

    redis = RedisBase(host="192.168.31.184", db=1)
    while 1:
        data = redis.read("Data")
        print(data)
        time.sleep(1)
