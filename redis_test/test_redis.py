import redis

def test_redis_connection(host, port=6379, db=1):
    try:
        r = redis.StrictRedis(host=host, port=port, db=db, socket_connect_timeout=5)
        r.ping()  # 测试连接
        print(f"✅ 成功连接到 Redis {host}:{port}")
        return True
    except Exception as e:
        print(f"❌ 无法连接 Redis {host}:{port} - 错误: {e}")
        return False

test_redis_connection("127.0.0.1")
test_redis_connection("192.168.31.179")  # 对比测试