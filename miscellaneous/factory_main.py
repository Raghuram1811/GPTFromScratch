from abc import ABC, abstractmethod

class Factory(ABC):
    @abstractmethod
    def show_product(self):
        pass

class ProductA(Factory):
    def show_product(self):
        print("This is Product A")

class ProductB(Factory):
    def show_product(self):
        print("This is Product B")  

class FactoryProducer:

    @classmethod
    def get_factory(cls, factory_type):
        if factory_type == "A":
            return ProductA()
        elif factory_type == "B":
            return ProductB()
        else:
            return None 

def main():

    factoryProducer = FactoryProducer()

    factoryA = factoryProducer.get_factory("A")
    factoryA.show_product()

    factoryB = factoryProducer.get_factory("B")
    factoryB.show_product()

if __name__ == "__main__":
    main()