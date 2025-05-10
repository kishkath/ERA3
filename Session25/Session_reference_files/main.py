import asyncio
import code

async def print_with_delay(message: str, delay: float) -> None:
    print(f"Starting : {message}")
    await asyncio.sleep(delay)
    print(f"Finished: {message}")

async def main():
    print("Starting main function...")

    task1 = asyncio.create_task(print_with_delay("Task 1", 2))
    task2 = asyncio.create_task(print_with_delay("Task 2", 1))
    task3 = asyncio.create_task(print_with_delay("Task 3", 3))
    code.interact(local=locals())

    await asyncio.gather(task1, task2, task3)

    print("All tasks completed!")

def add(a: int, b: int) -> int:
    return int(a+b)

def subtract(a: int, b: int) -> int:
    return int(a-b)

function_map = {'add': add, 'subtract': subtract}

# dotenv
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(api_key)

# decorators
def my_decorators(func):
    def wrapper():
        print("Before the function call")
        func()
        print("After the function call")
    return wrapper

@my_decorators
def say_hello():
    print(f"Hello, World!")

say_hello()


# asyncio with try-except

async def risky_operation(task_id: int, delay: float) -> None:
    try:
        print(f"Task {task_id}: Starting risky operation")
        await asyncio.sleep(delay)

        if task_id == 2:
            print(f"Task {task_id}: failed")
            raise ValueError(f"Task {task_id} failed!")

        print(f"Task {task_id}: Operation completed successfully")

    finally:
        print(f"Task {task_id}: Cleanup completed")

async def main():
    try:
        print("Starting main function...")

        tasks = [asyncio.create_task(risky_operation(1, 2)),
                 asyncio.create_task(risky_operation(2, 1)),
                 asyncio.create_task(risky_operation(3, 3))]

        await asyncio.gather(*tasks, return_exceptions=True)

    finally:
        print("Main function cleanup - Always executes !")

#
# if __name__ == "__main__":
#     # asyncio.run(main())

# if __name__ == "__main__":
#     print("Method 1: using dictionary access")
#     result1 = function_map['add'](5, 3)
#     print(f"5 + 3 = {result}")
#
#     result2 = function_map['subtract'](2, 3)
#     print(f" 2 ^ 3 = {result2}")