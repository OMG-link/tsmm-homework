import os
import sys
import random

def main():
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <n> <m> <id>")
        sys.exit(1)

    try:
        n = int(sys.argv[1])
        m = int(sys.argv[2])
        matrix_id = sys.argv[3]
    except ValueError:
        print("参数 n 和 m 应该是整数")
        sys.exit(1)

    matrix = [[round(random.uniform(0.0, 100.0), 2) for _ in range(m)] for _ in range(n)]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"{n}x{m}_{matrix_id}.txt"
    filepath = os.path.join(script_dir, filename)

    with open(filepath, 'w') as f:
        f.write(f"{n} {m}\n")
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")

if __name__ == "__main__":
    main()
