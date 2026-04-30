

if __name__ == "__main__":
    def f(**kwargs):
        print(kwargs.get("a"))

    f(a=1, b=2, c=3)