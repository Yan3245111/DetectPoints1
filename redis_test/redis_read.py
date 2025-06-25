
import redis
import msgpack

class RedisBase():
    def __init__(self):       
        self.__r = None
        
    def connect_to_db(self, host, database, socket_connect_timeout, port=6379):
        self.__r = redis.StrictRedis(host=host, port=port, db=database, socket_connect_timeout=socket_connect_timeout)

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
            # print(Exception, e)
            # print("data is not existed")
            return None


if __name__ == "__main__":
    redis_base = RedisBase()
    redis_base.connect_to_db(host="127.0.0.1", database=1, port=6379, socket_connect_timeout=1)
    while 1:
        data = redis_base.read("hello")
        print(data)
