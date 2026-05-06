from functools import lru_cache

@lru_cache(maxsize=128)
def fibo(n):
    if n<2:
        return n
    return fibo(n-1) + fibo(n-2)

def test_lambda():

    d = {'Apple': 34, 'Omange': 66, 'Banana': 10}
    d = sorted(d.items(), key=lambda x:x[1], reverse=True)
    print(d)

if __name__ == "__main__":
    print("This is a test file for some basic tests.")

    # print(fibo(50))

    test_lambda()


