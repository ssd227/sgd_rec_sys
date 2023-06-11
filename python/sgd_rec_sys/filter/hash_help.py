import random

# todo  比预期的误判率要高不少，难道是hash函数组选的不是很好，导致碰撞概率变大？
class BetterHashFunctions:
    def __init__(self, num_functions, m):
        self.num_functions = num_functions
        self.hash_functions = self.generate_hash_functions()
        self.m = m
        
    def __getitem__(self, id):
        return self.hash_functions[id]

    def generate_hash_functions(self):
        hash_functions = []
        random.seed(42)  # 设置随机种子，确保可重复性

        for _ in range(self.num_functions):
            seed = random.randint(1, 1000)  # 生成随机种子
            hash_functions.append(self.create_hash_function(seed))

        return hash_functions

    def create_hash_function(self, seed):
        random.seed(seed)  # 使用指定的种子生成随机数
        a = random.randint(1, 10000)
        b = random.randint(1, 10000)

        def hash_function(input):
            # 实现哈希函数的逻辑
            hash_value = (a * hash(input) + b) % self.m  # 使用哈希函数计算哈希值
            return hash_value

        return hash_function


class MyHashFunctions:
    def __init__(self, num_functions, m):
        self.num_functions = num_functions
        self.hash_functions = self.generate_hash_functions()
        self.m = m
        
    def __getitem__(self, id):
        return self.hash_functions[id]

    def generate_hash_functions(self):
        hash_functions = []
        random.seed(42)  # 设置随机种子，确保可重复性

        for _ in range(self.num_functions):
            seed = random.randint(1, 1000)  # 生成随机种子
            hash_functions.append(self.create_hash_function(seed))

        return hash_functions

    def create_hash_function(self, seed):
        random.seed(seed)  # 使用指定的种子生成随机数
        a = random.randint(1, 10000)
        b = random.randint(1, 10000)

        def hash_function(input):
            # 实现哈希函数的逻辑
            hash_value = (a * hash(input) + b) % self.m  # 使用哈希函数计算哈希值
            return hash_value

        return hash_function

# # 创建一个实例，生成5个不同的哈希函数
# num_functions = 5
# my_hash_functions = MyHashFunctions(num_functions)

# # 使用生成的哈希函数
# input_data = "example_data"
# for i in range(num_functions):
#     hash_value = my_hash_functions.hash_functions[i](input_data)
#     print(f"Hash value {i+1}: {hash_value}")
